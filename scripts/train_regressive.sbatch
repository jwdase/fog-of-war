#!/bin/bash
#SBATCH --job-name=fowhist-train
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jwdase@mit.edu

# Train src/regressive.py - the history model - on the same shards as
# scripts/train_model.sbatch trains src/model.py on.
#
# Why these numbers:
#
#   time        24:00 is the partition maximum, and --hours 24.0 is deliberately
#               equal to it rather than under it.  The gradient steps therefore
#               do *not* fit in one allocation: the RAM cache and torch.compile
#               come out of the front of it.  See "the requeue" below.
#   batch       1024 windows of 8 plies.  Two things fall out of that being
#               1024 x 8 = 8192 x 1: it is the 10M single-position run's step
#               token for token, and - because the time axis is causal, so every
#               ply in a window is a target - supervised square for supervised
#               square as well.  Measured 49 GB on an 80 GB card.  It is the
#               memory ceiling that picks this, as it did for the 10M and 50M
#               single-position runs; 2048 does not fit.
#   lr          7e-4, the 10M single-position run's, which is the point of
#               matching the supervised-square count above - one fewer thing
#               differing between the two runs.
#   history     8 plies, four of white's own moves.  The parameter count barely
#               moves with it but every activation does, so this is a throughput
#               dial, not a capacity one.
#   cpus        32.  A window gathers 8 plies per position, so the loader moves
#               8x the bytes per batch that src/train.py's does - but at ~1/20th
#               the batches per second, so it is not close to being the limit.
#               `data_wait_frac` in metrics.csv is the dial to watch.
#
# The requeue
# -----------
# --requeue alone does not do it.  SLURM requeues on preemption and node
# failure, not on a process that caught SIGUSR1 and exited 0, which is exactly
# what this job does five minutes before the walltime.  So src/train_regressive
# exits 42 when the clock ran out before the budget did, and the block at the
# bottom turns that into `scontrol requeue`.  The second allocation restores the
# elapsed clock along with the weights, so the cosine schedule spans the true 24
# hours rather than restarting, and only the last few minutes of it run there.
#
# If a second allocation is not worth the queue wait for those minutes, pass
# `--hours 23.5` and the run finishes inside the first one.
#
# Everything the run produces is under logs/fowhist-<jobid>/ - see the module
# docstring in src/train.py for what each file is.  Watch it with:
#
#     tail -f logs/fowhist-train-<jobid>.out
#     column -s, -t logs/fowhist-<jobid>/val.csv | less -S

set -euo pipefail

# sbatch exports the submitting shell's environment, so submitting this from
# inside another allocation carries that allocation's CPU mask in as
# SLURM_CPU_BIND, and srun then tries to bind to cores this job was never
# given.  That is what killed job 23050365.
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE

REPO=/home/jwdase/projects/fog-of-war
cd "$REPO"

export FOW_PROCESSED=${FOW_PROCESSED:-/home/jwdase/orcd/pool/fog-of-war/processed}
export PYTHONPATH="$REPO"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/torchinductor-$USER

# The history model holds ~49 GB of activations and frees them in a different
# order than it allocates them; without this the allocator ends a long run with
# enough reserved-but-unallocated memory to OOM on a step it has already run
# thousands of times.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} - 8 ))

echo "job        : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "restarts   : ${SLURM_RESTART_COUNT:-0}"
echo "gpu        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "cores      : ${SLURM_CPUS_PER_TASK:-?} allocated, $WORKERS loader workers"
echo "shards     : $FOW_PROCESSED"
echo "started    : $(date)"
echo

mkdir -p logs

set +e
srun .venv/bin/python -m src.train_regressive \
    --shards "$FOW_PROCESSED" \
    --workers "$WORKERS" \
    --hours 24.0 \
    --history 8 \
    --batch-size 1024 \
    --lr 7e-4 \
    --name fowhist \
    --resume auto \
    "$@"
CODE=$?
set -e

echo
echo "finished   : $(date) (exit $CODE)"

# 42 is src/train_regressive.UNFINISHED: the walltime ran out before --hours
# did.  last.pt is already written, so requeueing costs a cache fill and a
# compile and nothing else.
if [ "$CODE" -eq 42 ]; then
    echo "budget not spent - requeueing ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi

exit "$CODE"
