#!/bin/bash
#SBATCH --job-name=fow-train
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --time=13:00:00
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

# Train src/model.py on the shards src/data.py wrote.
#
# Why these numbers:
#
#   partition   ou_bcs_normal is the only non-preempting partition open to this
#               account that has H100s and a walltime past 12h (mit_normal_gpu
#               caps at 6h, mit_preemptable reaches 48h but can requeue the job
#               out from under us).  It has 5 nodes of gpu:h100:4.
#   gpu         one.  The model is 20k parameters; a second GPU would spend more
#               time synchronising gradients than computing them.  What is
#               actually scarce here is CPU, not GPU.
#   cpus        32.  Decompressing a shard is ~0.35s of one core and yields
#               ~83,000 positions, so a core feeds ~240k positions/s and the
#               H100 will take several million.  This is the dial to turn if
#               `data_wait_frac` in metrics.csv is not near zero.
#   mem         256G.  The whole compressed corpus is ~21 GB and is held in RAM
#               for the length of the run, so the shared filesystem is read once
#               instead of once per pass; the rest is loader workers, each
#               holding a few decompressed shards.
#   time        13:00 against a --hours 12.0 budget, so the 12 hours are 12
#               hours of gradient steps.  The hour of slack covers filling the
#               RAM cache (a few minutes for 21 GB), torch.compile, and the
#               final validation pass.
#   signal      USR1 five minutes before the walltime.  src/train.py catches it,
#               finishes the step, writes last.pt and exits, so --requeue picks
#               the run up where it stopped rather than at step 0.
#
# Everything the run produces is under logs/fow-<jobid>/ - see the module
# docstring in src/train.py for what each file is.  Watch a running job with:
#
#     tail -f logs/fow-train-<jobid>.out
#     column -s, -t logs/fow-<jobid>/metrics.csv | less -S

set -euo pipefail

# sbatch exports the submitting shell's environment, so submitting this from
# inside another allocation - an interactive `srun --pty bash`, say - carries
# that allocation's CPU mask in as SLURM_CPU_BIND.  The srun below then tries
# to bind this job's task to cores on a node it was never given, and the job
# dies in a second with "Unable to satisfy cpu bind request".  The binding this
# job wants is the one SLURM works out for it.
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE

REPO=/home/jwdase/projects/fog-of-war
cd "$REPO"

# Where src/data.py put the shards.
export FOW_PROCESSED=${FOW_PROCESSED:-/home/jwdase/orcd/pool/fog-of-war/processed}
export PYTHONPATH="$REPO"

# The loader is many processes doing one shard each; a BLAS pool per worker
# would only have them fighting over the same cores.  src/preprocess.py pins
# each worker to one thread as well, for anything that imports numpy early.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Keep the compile cache off the shared filesystem and out of everyone's way.
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/torchinductor-$USER

# One loader worker per core, less the ones the main process and the GPU feed
# needs for themselves.
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} - 8 ))

echo "job        : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "gpu        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "cores      : ${SLURM_CPUS_PER_TASK:-?} allocated, $WORKERS loader workers"
echo "shards     : $FOW_PROCESSED"
echo "started    : $(date)"
echo

mkdir -p logs

srun .venv/bin/python -m src.train \
    --shards "$FOW_PROCESSED" \
    --workers "$WORKERS" \
    --hours 12.0 \
    --batch-size 16384 \
    --lr 3e-3 \
    --name fow \
    --resume auto \
    "$@"

echo
echo "finished   : $(date)"
