export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

SCRIPT="benchmarks/run_clm.py"
OUTPUT_DIR=/flash/project_462000007/mvsjober/run-clm
DS_CONFIG=benchmarks/ds_config_clm.json

export HF_HOME=/scratch/project_462000007/mvsjober/hf-home
export TORCH_HOME=/scratch/project_462000007/mvsjober/torch-cache

if [ "$SLURM_NTASKS" -ne "$SLURM_NNODES" ]; then
    echo "ERROR: this script needs to be run as one task per node."
    echo "SLURM_NNODES = $SLURM_NNODES != SLURM_NTASKS = $SLURM_NTASKS"
    exit 1
fi


if [ "$SLURM_NNODES" -gt 1 ]; then
     RDZV_HOST=$(hostname)
     RDZV_PORT=29400

     (set -x
      srun python3 -m torch.distributed.run --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
           --nnodes=$SLURM_NNODES --nproc_per_node=$NUM_GPUS $SCRIPT \
           --deepspeed $DS_CONFIG \
           --model_name_or_path EleutherAI/gpt-neo-1.3B \
           --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
           --per_device_train_batch_size 2 --do_train \
           --output_dir $OUTPUT_DIR --overwrite_output_dir \
           --num_train_epochs 1 --dataloader_num_workers 7 $SCRIPT_OPTS $*
     )
else
    if [ ! -z $NUM_GPUS ]; then
        SCRIPT_OPTS="--gradient_accumulation_steps $(( 8 / $NUM_GPUS ))"
    fi

    (set -x
     srun singularity_wrapper exec deepspeed --num_gpus=$NUM_GPUS $SCRIPT --deepspeed $DS_CONFIG \
          --model_name_or_path EleutherAI/gpt-neo-1.3B \
          --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
          --per_device_train_batch_size 2 --do_train \
          --output_dir $OUTPUT_DIR --overwrite_output_dir \
          --num_train_epochs 1 --dataloader_num_workers 7 $SCRIPT_OPTS $*
    )
fi

rm -rf $OUTPUT_DIR
