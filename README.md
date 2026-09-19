
# Location of Processed Data
```
/home/jwdase/orcd/pool/fog-of-war/processed
```

# Predicting the hidden board

`src/model.py`, `src/preprocess.py` and `src/train.py` train a network to answer
`P(a piece is on a square)` from white's point of view, given only what white can
see.

## The model

64 tokens, one per square, each the sum of the four encoders:

| encoder        | values                                               |
| -------------- | ---------------------------------------------------- |
| `e_piece`      | empty, pawn, bishop, knight, rook, queen, king       |
| `e_square`     | one of 64, learned                                   |
| `e_visibility` | `0` hidden, `1` visible                              |
| `e_side`       | `+1` white, `-1` black (`0` for an empty square)      |

Three pre-norm transformer layers over those tokens, then a 13-way softmax per
square - empty, white's six, black's six - so a query is one entry of the
output:

```python
probabilities = model.predict(piece, side, visibility)
probability_of(probabilities, 'a7', game.PAWN, -1)   # P(black pawn on a7)
```

**49,205 parameters**: 3,040 in the embeddings, 15,184 per layer, 613 in the
final norm and head. `python -m src.model` prints the breakdown.

The budget went into depth rather than width - `d_model=40`, three layers -
because working out what is behind the fog is a multi-hop argument and each
layer is one round of squares asking each other questions.

Note that `e_side` writes white as `+1`, which is the opposite of the sign
convention inside `src/game.py`, where a board array writes white *negative*.
`src/preprocess.py` is the only place the two meet.

## The data

`src/data.py` saved what white could see but not *where it could see*, and a `0`
on the board means both "visibly empty" and "hidden".
`preprocess.visibility_mask` rebuilds the missing mask as a vectorised
ray-march - the same pseudo-legal-move rule `ChessState.visible_squares` uses,
en passant included - and `tests/test_preprocess.py` checks it square for square
against `ChessState` on real games, then checks `truth * mask == view` on a
whole shard. `src/train.py` re-runs the second check at the start of every run.

The ~21 GB of compressed shards are held in RAM for the length of a run, so the
shared filesystem is read once rather than once per pass.

## Running it

```bash
# End to end on a login node - a few shards, 30 steps, no RAM cache.
python -m src.train --smoke --device cpu

# The real thing: 12h of gradient steps on one H100, in a 13h allocation.
sbatch scripts/train_model.sbatch
```

Everything a run produces lands in `logs/<name>-<jobid>/`: `train.log`,
`metrics.csv`, `val.csv`, `progress.json` (latest numbers only, for watching a
running job), `config.json`, `env.json` and `checkpoints/{last,best}.pt`. The
job catches `SIGUSR1` from SLURM, checkpoints, and `--resume auto` picks the run
up where it stopped, so a requeue costs a few minutes rather than the run.

### Reading the numbers

`acc` over all 64 squares is not the metric - white can see about 45 of them and
copying its own input is free. The `hidden_*` columns are the task, and
`baseline` is what to compare them against: the accuracy of answering "empty"
for every hidden square. `hidden_piece_acc`, over the hidden squares that
actually hold something, is the hard half.
