'''The history model: every board white has seen this game in, one belief out.

:class:`~src.model.FogOfWarNet` answers from a single position, which throws
away the thing a fog-of-war player actually reasons from.  A rook seen on a8
twelve plies ago and never seen since is *probably still near a8* - it had to
move somewhere, and white would likely have seen it go.  A square that white
watched empty itself three plies ago is empty for a different reason than one
it has never seen.  None of that is recoverable from the current view, and it
is exactly where the single-position model is weakest: rooks, whose whole
nature is to sit still on a file for twenty plies and then cross the board,
come back at 0.32 top-1 against 0.63 for the king.

So this model eats ``--history`` consecutive views, oldest to newest, and
predicts the true board at *every* one of them.

Why every one, and not just the newest
--------------------------------------

Predicting only the newest ply is the obvious reading of the task and it wastes
seven eighths of the forward pass: eight plies go in, one square-grid of
gradient comes out.  Measured on an A100 that is 147k supervised squares a
second against 3.35M for the single-position 10M model - the history model
would see an order of magnitude less of the corpus for the same day of GPU, and
a comparison between them would be measuring the data budget rather than the
architecture.

Making the time axis *causal* fixes it.  Ply ``k`` may attend to plies ``0..k``
and no further, so its prediction is conditioned only on what white had
actually seen by then - which makes every ply in the window a legitimate
training target, not just the last.  The supervised squares per step go back up
by ``T``, to exactly the single-position model's, and the deployment query is
unchanged: the newest ply attends to the whole window, because the whole window
is its past.

Shape of the argument
---------------------

The tokens are a ``(time, square)`` grid - ``T x 64`` of them - and the useful
questions run along both axes:

*along squares*  "the rest of the board looks like this, so where can the queen
                 be" - what the single-position model already does.
*along time*     "what was on this square before, and how long ago" - the new
                 information.

Attending over all ``T * 64`` tokens at once would ask both at once, and cost
``(T * 64)^2``: at ``T=8`` that is sixty-four times the single-position model's
attention for eight times the input.  So each block factorises, the way a video
transformer does - attention across the 64 squares of one ply, then attention
across the T plies of one square, then the feed-forward.  That costs
``T * 64^2 + 64 * T^2``, which is linear in the history length, and two
factorised rounds reach any (time, square) pair the full attention would.  Six
blocks is twelve rounds.

``src/train_regressive.py`` trains on all ``T`` grids and validates on the
newest alone, so its numbers stay comparable with the single-position runs.

What it is still not
--------------------

Sixty-four independent softmaxes, exactly as before.  History is *information*,
not *coherence*: this model should know much better where the rook is, and will
still happily put 0.6 of a rook on two different squares, because nothing in
the parameterisation couples them.  Sampling whole boards needs a different
head - see the note in ``src/train_regressive.py``.

**9,898,573 parameters** at the defaults, which is within 0.1% of the 10M
single-position run (9,892,813) so the two are a fair comparison.  ``python -m
src.regressive`` prints the breakdown.
'''

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import nn

from src.model import (
    N_CLASSES,
    N_PIECE_TYPES,
    N_SIDES,
    N_SQUARES,
    N_VISIBILITY,
)


#: Plies of history the defaults feed in, counting the current one.  Eight plies
#: is four of white's own moves: long enough that a piece which stepped out of
#: sight is still in the window, short enough that the ~8x token cost over the
#: single-position model is affordable inside a day.
DEFAULT_HISTORY = 8


@dataclass(frozen=True)
class RegressiveConfig:
    '''Shape of the network.

    ``history`` is the one dial here that is not free: every token count, every
    activation and the whole step time scale with it, while the parameter count
    barely moves (it buys one row of ``e_time``).  Widening the window is a
    throughput decision, not a capacity one.
    '''

    d_model: int = 320
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1280
    dropout: float = 0.0
    history: int = DEFAULT_HISTORY

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError(f'{self.d_model} channels do not divide into {self.n_heads} heads')
        if self.history < 1:
            raise ValueError(f'history has to be at least the current ply, got {self.history}')

    def to_dict(self):
        return asdict(self)


class HistoryEncoder(nn.Module):
    '''The four square embeddings of :class:`~src.model.SquareEncoder`, plus time.

    ``e_time`` is indexed by *plies ago* - 0 is the ply being predicted, T-1 the
    oldest in the window - rather than by absolute ply number.  Recency is what
    the argument turns on ("I saw it two moves ago") and a relative index means
    ply 40 of a long game looks like ply 8 of a short one, so the table is
    learned from every position in the corpus instead of from the few games that
    reach a given length.
    '''

    def __init__(self, config):
        super().__init__()
        self.e_piece = nn.Embedding(N_PIECE_TYPES, config.d_model)
        self.e_square = nn.Embedding(N_SQUARES, config.d_model)
        self.e_visibility = nn.Embedding(N_VISIBILITY, config.d_model)
        self.e_side = nn.Embedding(N_SIDES, config.d_model)
        self.e_time = nn.Embedding(config.history, config.d_model)

        self.register_buffer(
            'squares', torch.arange(N_SQUARES, dtype=torch.long), persistent=False
        )
        # arange(T-1, -1, -1): the last slot is the present, so it gets 0.
        self.register_buffer(
            'ago',
            torch.arange(config.history - 1, -1, -1, dtype=torch.long),
            persistent=False,
        )

    def forward(self, piece, side, visibility):
        '''``(batch, T, 64)`` integers in, ``(batch, T, 64, d_model)`` out.'''
        return (
            self.e_piece(piece)
            + self.e_side(side)
            + self.e_visibility(visibility)
            + self.e_square(self.squares)[None, None, :, :]
            + self.e_time(self.ago)[None, :, None, :]
        )


class Attention(nn.Module):
    '''Bidirectional multi-head attention over whichever axis the caller folded.

    The same module serves both axes: the block hands it ``(rows, tokens,
    d_model)`` having already moved the axis it wants attended into the middle,
    so nothing here needs to know whether ``tokens`` means squares or plies.

    ``causal`` is what the time axis wants and the space axis must not have: a
    ply may see the plies before it, a square sees every other square.

    The two axes take different code paths, and both halves of that were forced
    by something that went wrong first.

    The space axis uses ``scaled_dot_product_attention``: 64 tokens is long
    enough for the flash kernel to be worth having, and with no mask it gets
    one.

    The time axis forms its scores itself, for two reasons.  Passing an
    explicit ``attn_mask`` to SDPA knocks it onto the memory-efficient kernel,
    whose *backward* asked for 16 GiB on a batch of 1024 windows.  Asking
    instead for ``is_causal=True`` keeps the flash kernel but cannot launch it:
    the time axis folds the 64 squares into the batch, so a batch of 1024 is
    65,536 rows, and CUDA caps a grid dimension at 65,535 - one short, and it
    fails as ``invalid argument`` rather than as anything that names the cause.
    Eight plies of scores are 33 MB, so materialising them sidesteps both and
    costs nothing worth measuring.
    '''

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

        # Only the time axis uses this, and it is (history x history) of bool.
        self.register_buffer(
            'causal_mask',
            torch.ones(config.history, config.history, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x, causal=False):
        rows, tokens, _ = x.shape

        qkv = self.qkv(x).view(rows, tokens, 3, self.n_heads, self.d_head)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)

        if not causal:
            attended = F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            scores = (query @ key.transpose(-1, -2)) * self.d_head ** -0.5
            scores = scores.masked_fill(
                ~self.causal_mask[:tokens, :tokens], float('-inf')
            )

            # Under autocast the scores are bf16; softmax is on torch's fp32
            # list, so the normalisation itself is not done in 8 bits.
            weights = scores.softmax(-1)
            if self.training and self.dropout:
                weights = F.dropout(weights, self.dropout)

            attended = weights @ value

        return self.proj(attended.transpose(1, 2).reshape(rows, tokens, -1))


class SpaceTimeBlock(nn.Module):
    '''Squares talk, then plies talk, then the feed-forward - each pre-norm.

    Space first so that a ply is understood as a position before it is compared
    with the plies around it: what makes "the rook left a8" legible is that the
    model has already read both boards as boards.
    '''

    def __init__(self, config):
        super().__init__()
        self.norm_space = nn.LayerNorm(config.d_model)
        self.space = Attention(config)

        self.norm_time = nn.LayerNorm(config.d_model)
        self.time = Attention(config)

        self.norm_feedforward = nn.LayerNorm(config.d_model)
        self.up = nn.Linear(config.d_model, config.d_ff)
        self.down = nn.Linear(config.d_ff, config.d_model)

        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, plies, squares, channels = x.shape

        # ---- across the 64 squares of each ply -------------------------
        folded = self.norm_space(x).reshape(batch * plies, squares, channels)
        x = x + self.drop(self.space(folded).view(batch, plies, squares, channels))

        # ---- across the T plies of each square -------------------------
        # (batch, ply, square, c) -> (batch, square, ply, c), so that the axis
        # attention runs over is the middle one.  transpose(1, 2) leaves the
        # tensor non-contiguous and .reshape would silently copy it, so the
        # copy is asked for once, here, rather than hidden in a view.
        folded = self.norm_time(x).transpose(1, 2).contiguous()
        folded = folded.view(batch * squares, plies, channels)

        attended = self.time(folded, causal=True).view(batch, squares, plies, channels)
        x = x + self.drop(attended.transpose(1, 2))

        # ---- and the feed-forward, per token ---------------------------
        return x + self.drop(self.down(F.gelu(self.up(self.norm_feedforward(x)))))


class RegressiveFogOfWar(nn.Module):
    '''T plies of white's view in, 64 distributions over 13 classes out.

    Inputs are three ``(batch, T, 64)`` integer tensors - ``piece``, ``side``,
    ``visibility`` - oldest ply first, newest last.  Output is ``(batch, T, 64,
    13)``: one 64-square belief per ply, each conditioned on that ply and the
    ones before it.

    ``logits[:, -1]`` is the query this model exists to answer and is
    interchangeable with :class:`~src.model.FogOfWarNet`'s output, so every
    metric in ``src/train.py`` reads it unchanged.
    '''

    def __init__(self, config=None):
        super().__init__()
        self.config = config or RegressiveConfig()

        self.encoder = HistoryEncoder(self.config)
        self.blocks = nn.ModuleList(
            SpaceTimeBlock(self.config) for _ in range(self.config.n_layers)
        )
        self.norm = nn.LayerNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, N_CLASSES)

        self.apply(self._init)

        # Three writes into the residual stream per block now rather than two,
        # so the projections that make them start smaller by the same rule.
        for name, parameter in self.named_parameters():
            if name.endswith(('space.proj.weight', 'time.proj.weight', 'down.weight')):
                nn.init.normal_(parameter, std=0.02 / (3 * self.config.n_layers) ** 0.5)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, piece, side, visibility):
        '''``(batch, T, 64, 13)`` logits - one belief per ply in the window.'''
        x = self.encoder(piece, side, visibility)

        for block in self.blocks:
            x = block(x)

        return self.head(self.norm(x))

    @torch.no_grad()
    def predict(self, piece, side, visibility):
        '''``(batch, 64, 13)`` probabilities for the newest ply, softmaxed.

        The older plies are dropped here: they are a training signal, not an
        answer anyone asked for.  Read the whole window with :meth:`forward`.
        '''
        self.eval()
        return F.softmax(self.forward(piece, side, visibility)[:, -1].float(), dim=-1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_breakdown(model):
    '''Where the parameters went, in the four groups worth naming.'''
    groups = {'encoder': 0, 'blocks': 0, 'norm': 0, 'head': 0}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        top = name.split('.')[0]
        groups[top if top in groups else 'blocks'] += parameter.numel()

    return groups


if __name__ == '__main__':
    model = RegressiveFogOfWar()
    groups = parameter_breakdown(model)

    print(f'config: {model.config}')
    print(f'parameters: {parameter_count(model):,}')
    for name, count in groups.items():
        print(f'  {name:<10} {count:>12,}')
    print(f'  {"per block":<10} {groups["blocks"] // model.config.n_layers:>12,}')
