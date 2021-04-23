module list

if [ -n "$DATA_TAR" ]; then
    echo "Extracting $DATA_TAR to $LOCAL_SCRATCH"
    (set -x
     srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
          tar xf $DATA_TAR -C $LOCAL_SCRATCH
    )
fi

set -x

date
hostname
nvidia-smi

