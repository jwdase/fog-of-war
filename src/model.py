'''The network: white's view of a fog-of-war position in, a belief about every
square out.

The board is 64 tokens, one per square, and every token is the sum of the four
embeddings the task is described by:

``e_piece``
    what is standing there *as far as white can tell* - one of ``EMPTY, PAWN,
    BISHOP, KNIGHT, ROOK, QUEEN, KING``.  A square white cannot see reads
    ``EMPTY``, which is the whole difficulty: ``e_visibility`` is what separates
    "empty" from "unknown".

``e_square``
    which of the 64 squares this is.  A learned table rather than a coordinate,
    because the board is small enough to memorise and the useful structure -
    back rank, centre, the files a rook is likely to be on - is not linear in
    (row, column).

``e_visibility``
    1 where white can see, 0 where it cannot.

``e_side``
    whose piece it is.  Written ``+1`` for white and ``-1`` for black as
    :data:`SIDE_SIGN` says, stored as the table indices :data:`SIDE_NONE`,
    :data:`SIDE_WHITE`, :data:`SIDE_BLACK` because an embedding is a lookup.

Attention over those 64 tokens is the point of the architecture.  What is hidden
on one square is almost entirely determined by what is visible on the others -
a piece that has to be *somewhere*, a rank that has been emptied, a bishop that
can only be on one colour - and attention is how a square gets to ask.

The output is a 13-way distribution per square: empty, six white pieces, six
black ones.  So ``P(a pawn is on a7)`` is one softmax entry, which is what
:func:`probability_of` reads out.  A softmax rather than 12 independent
sigmoids because a square holds exactly one thing, and normalising over that
fact is free information.

Size is a design constraint here, not an accident: the default configuration is
49,205 parameters, and :func:`parameter_count` is what keeps it honest.
'''

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------
# The vocabulary the model eats
#
# src/preprocess.py produces exactly these indices; nothing else may invent
# them.  They are here rather than there because they are the model's input
# contract - what the embedding tables are sized from.
# --------------------------------------------------------------------------

#: Squares on a board, in the order src/game.py stores them: index ``row * 8 +
#: column``, row 0 == rank 8, column 0 == file a.  So ``a8`` is 0 and ``a7`` is
#: 8 - see :func:`square_index`.
N_SQUARES = 64

#: ``e_piece``: empty plus the six piece codes of ``src/game.py``, unsigned.
N_PIECE_TYPES = 7

#: ``e_visibility``: hidden, visible.
N_VISIBILITY = 2
HIDDEN, VISIBLE = 0, 1

#: ``e_side``: no piece, white, black.
N_SIDES = 3
SIDE_NONE, SIDE_WHITE, SIDE_BLACK = 0, 1, 2

#: What the side indices mean as the signed quantity they stand for.  White is
#: ``+1`` and black ``-1`` here.  Note this is the opposite of the sign
#: convention inside ``src/game.py``, where a *board array* writes white
#: negative; ``src/preprocess.py`` is where the two meet.
SIDE_SIGN = {SIDE_NONE: 0, SIDE_WHITE: +1, SIDE_BLACK: -1}

#: Piece names by code, following ``src/game.py`` - note ``B`` is 2 and ``N`` is
#: 3, which is the other way round from python-chess.
PIECE_NAMES = ('.', 'P', 'B', 'N', 'R', 'Q', 'K')

#: The 13 things a square can be: empty, white's six, black's six.
N_CLASSES = 13
CLASS_EMPTY = 0
CLASS_NAMES = (
    '.',
    'P', 'B', 'N', 'R', 'Q', 'K',
    'p', 'b', 'n', 'r', 'q', 'k',
)


def class_index(piece, side):
    '''The output class for ``piece`` (a ``src/game.py`` code) held by ``side``.

    ``side`` is a :data:`SIDE_SIGN` value - ``+1`` white, ``-1`` black - or one
    of the ``SIDE_*`` indices, which are distinguishable because ``-1`` is not
    one of them.
    '''
    if piece == 0:
        return CLASS_EMPTY

    if side in (SIDE_WHITE, +1):
        return piece
    if side in (SIDE_BLACK, -1):
        return 6 + piece

    raise ValueError(f'a {PIECE_NAMES[piece]} has to belong to someone, got {side!r}')


def square_index(name):
    '''The token position of a square named the way people name them: ``'a7'``.'''
    file_, rank = name[0].lower(), int(name[1])
    return (8 - rank) * 8 + (ord(file_) - ord('a'))


#: Square names in token order, so ``SQUARE_NAMES[8] == 'a7'``.
SQUARE_NAMES = tuple(
    f'{chr(ord("a") + column)}{8 - row}' for row in range(8) for column in range(8)
)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    '''Shape of the network.

    The defaults are chosen to land on ~50k parameters, which is the budget this
    model is built to: 3040 in the four embedding tables, 15,184 per encoder
    layer, 613 in the final norm and head.  ``d_model`` is the expensive dial -
    the layers grow with its square - and ``n_layers`` is the cheap one.

    The budget went into depth rather than width.  Working out what is behind
    the fog is a multi-hop argument - this square is empty, so that piece is
    somewhere else, and it has to be *somewhere* - and each layer is one round
    of squares asking each other questions.  Three rounds at ``d_model=40``
    beats two at 48 for the same parameters, or that is the bet; ``--n-layers``
    and ``--d-model`` are what an ablation would turn.
    '''

    d_model: int = 40
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 104
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError(f'{self.d_model} channels do not divide into {self.n_heads} heads')

    def to_dict(self):
        return asdict(self)


class SquareEncoder(nn.Module):
    '''The four embeddings of the module docstring, added into one token each.

    Added rather than concatenated so that every feature gets the full width to
    write into and the layers that follow do not have to spend parameters
    mixing four narrow fields back together.
    '''

    def __init__(self, config):
        super().__init__()
        self.e_piece = nn.Embedding(N_PIECE_TYPES, config.d_model)
        self.e_square = nn.Embedding(N_SQUARES, config.d_model)
        self.e_visibility = nn.Embedding(N_VISIBILITY, config.d_model)
        self.e_side = nn.Embedding(N_SIDES, config.d_model)

        self.register_buffer(
            'squares', torch.arange(N_SQUARES, dtype=torch.long), persistent=False
        )

    def forward(self, piece, side, visibility):
        return (
            self.e_piece(piece)
            + self.e_side(side)
            + self.e_visibility(visibility)
            + self.e_square(self.squares)
        )


class SelfAttention(nn.Module):
    '''Ordinary bidirectional multi-head attention over the 64 squares.

    No mask: every square may look at every other one, which is the only way a
    hidden square learns anything, since by definition it has nothing of its own
    to go on.
    '''

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        batch, tokens, _ = x.shape

        qkv = self.qkv(x).view(batch, tokens, 3, self.n_heads, self.d_head)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)

        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0
        )

        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, -1))


class Block(nn.Module):
    '''Pre-norm attention then pre-norm feed-forward, each added back in.

    Pre-norm because it trains at a high learning rate without a warmup long
    enough to matter, which is what a 12-hour budget wants.
    '''

    def __init__(self, config):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.attention = SelfAttention(config)

        self.norm_feedforward = nn.LayerNorm(config.d_model)
        self.up = nn.Linear(config.d_model, config.d_ff)
        self.down = nn.Linear(config.d_ff, config.d_model)

        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.drop(self.attention(self.norm_attention(x)))
        x = x + self.drop(self.down(F.gelu(self.up(self.norm_feedforward(x)))))
        return x


class FogOfWarNet(nn.Module):
    '''White's information set in, 64 distributions over 13 classes out.

    Inputs are three ``(batch, 64)`` integer tensors - ``piece``, ``side``,
    ``visibility`` - laid out in the square order of :data:`SQUARE_NAMES`.  The
    square index itself is not passed: it is the position in the sequence.
    '''

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()

        self.encoder = SquareEncoder(self.config)
        self.blocks = nn.ModuleList(Block(self.config) for _ in range(self.config.n_layers))
        self.norm = nn.LayerNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, N_CLASSES)

        self.apply(self._init)

        # The residual stream is written to twice per block, so the two
        # projections that write into it start proportionally smaller.
        for name, parameter in self.named_parameters():
            if name.endswith(('attention.proj.weight', 'down.weight')):
                nn.init.normal_(parameter, std=0.02 / (2 * self.config.n_layers) ** 0.5)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, piece, side, visibility):
        '''``(batch, 64, 13)`` logits.'''
        x = self.encoder(piece, side, visibility)

        for block in self.blocks:
            x = block(x)

        return self.head(self.norm(x))

    @torch.no_grad()
    def predict(self, piece, side, visibility):
        '''``(batch, 64, 13)`` probabilities - the forward pass, softmaxed.'''
        self.eval()
        return F.softmax(self.forward(piece, side, visibility).float(), dim=-1)


def probability_of(probabilities, square, piece, side):
    '''``P(side's piece is on square)``, read out of :meth:`FogOfWarNet.predict`.

    The query the model exists to answer::

        >>> probability_of(model.predict(*batch), 'a7', game.PAWN, -1)

    is the probability of a black pawn on a7, one number per row of the batch.
    '''
    return probabilities[..., square_index(square), class_index(piece, side)]


def parameter_count(model, trainable_only=True):
    '''How many parameters the thing has, for the budget this model lives under.'''
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


def parameter_breakdown(model):
    '''``{top-level module: parameter count}``, which is what a budget is spent on.'''
    counts = {}

    for name, parameter in model.named_parameters():
        counts.setdefault(name.split('.')[0], 0)
        counts[name.split('.')[0]] += parameter.numel()

    return counts


if __name__ == '__main__':
    net = FogOfWarNet()
    print(net.config)
    print(f'parameters: {parameter_count(net):,}')

    for part, count in parameter_breakdown(net).items():
        print(f'  {part:<10} {count:>8,}')
