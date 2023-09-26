export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

SCRIPT="benchmarks/run_clm.py"
OUTPUT_DIR=/flash/project_462000007/mvsjober/run-clm

export HF_HOME=/scratch/project_462000007/mvsjober/hf-home
export TORCH_HOME=/scratch/project_462000007/mvsjober/torch-cache


if [ "$SLURM_NNODES" -gt 1 ]; then
    if [ $(( $NUM_GPUS * $SLURM_NNODES )) -ne $SLURM_NTASKS ]; then
        echo "ERROR: this script needs to be run as one task per GPU. Try using slurm/*-mpi.sh scripts."
        echo "NUM_GPUS * SLURM_NNODES = $NUM_GPUS * $SLURM_NNODES != SLURM_NTASKS = $SLURM_NTASKS"
        exit 1
    fi
    
    (set -x
     srun python3 $SCRIPT --deepspeed benchmarks/ds_config_clm.json \
          --model_name_or_path EleutherAI/gpt-neo-1.3B \
          --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
          --per_device_train_batch_size 2 --do_train \
          --output_dir $OUTPUT_DIR --overwrite_output_dir \
          --gradient_accumulation_steps 1 \
          --num_train_epochs 1 --dataloader_num_workers 7 $SCRIPT_OPTS $*
     )
else
    if [ $SLURM_NTASKS -ne 1 ]; then
        echo "ERROR: single node runs need to be run as a single task"
        echo "SLURM_NTASKS = $SLURM_NTASKS != 1"
        exit 1
    fi

    if [ ! -z $NUM_GPUS ]; then
        SCRIPT_OPTS="--gradient_accumulation_steps $(( 8 / $NUM_GPUS ))"
    fi

    (set -x
     srun singularity_wrapper exec deepspeed --num_gpus=$NUM_GPUS $SCRIPT --deepspeed benchmarks/ds_config_clm.json \
          --model_name_or_path EleutherAI/gpt-neo-1.3B \
          --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
          --per_device_train_batch_size 2 --do_train \
          --output_dir $OUTPUT_DIR --overwrite_output_dir \
          --num_train_epochs 1 --dataloader_num_workers 7 $SCRIPT_OPTS $*
    )
fi

rm -rf $OUTPUT_DIR
