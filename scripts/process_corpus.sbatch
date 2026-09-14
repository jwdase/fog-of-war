#!/bin/bash
#SBATCH --job-name=fow-process
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jwdase@mit.edu

# Play the whole lichess-elite corpus into fog-of-war shards, one core per
# parquet file.
#
# Why these numbers:
#
#   partition   ou_bcs_normal is the only partition open to this account whose
#               walltime reaches 24h and that does not preempt.  mit_normal
#               caps at 12h; mit_preemptable allows 48h but can requeue the job
#               out from under us.
#   cpus        27 parquet files, one worker each, plus one for the parent that
#               collects their statistics.  src/parrallel_games.py pins each
#               worker to a core and holds every BLAS/OpenMP pool to one
#               thread, so the cores have to be real rather than oversubscribed.
#   time        ~5.8 h/file measured at 47 games/s/core on a login node.  All
#               27 run side by side, so the wall clock is one file's worth plus
#               slack for a busier node.
#   mem         a worker holds one file's `moves` column (~0.5 GB as arrow) plus
#               its shard buffers.  4 GB/core is generous against that.
#
# Output is ~21 GB for the whole 26.3M-game corpus, in ~26,000 shards.  That is
# not a typo: the raw tensors are ~1.2 TB, and src/data.py compresses them ~45x
# on the way out because a board barely changes between plies.  It goes to pool
# rather than HOME (only ~76 GB free) or scratch (purged).
#
# The run is resumable: a parquet file that already has shards in the output
# directory is skipped, so requeueing after a failure picks up where it left
# off.  Pass --overwrite to redo them.

set -euo pipefail

REPO=/home/jwdase/projects/fog-of-war
cd "$REPO"

# Where the shards land.
OUTPUT=${FOW_OUTPUT:-/home/jwdase/orcd/pool/fog-of-war/processed}

# One thread per worker.  parrallel_games.py sets these too, but a child that
# imports numpy before that block would miss them.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="$REPO"

# One worker per file; the parent keeps the spare core.
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-28} - 1 ))

echo "job        : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "cores      : ${SLURM_CPUS_PER_TASK:-?} allocated, $WORKERS workers"
echo "output     : $OUTPUT"
echo "started    : $(date)"
echo

mkdir -p "$OUTPUT"

srun .venv/bin/python -m src.parrallel_games \
    --workers "$WORKERS" \
    --output "$OUTPUT"

echo
echo "finished   : $(date)"
