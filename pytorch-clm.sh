export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

SCRIPT="benchmarks/run_clm.py"
OUTPUT_DIR=/flash/project_462000007/mvsjober/run-clm

export HF_HOME=/scratch/project_462000007/mvsjober/hf-home
export TORCH_HOME=/scratch/project_462000007/mvsjober/torch-cache

#DIST_OPTS="--standalone --master_port 0"
#SCRIPT_OPTS="--warmup-steps 1000 --workers=7"
#SCRIPT_OPTS="--warmup-steps 100 --workers=0"

if [ "$SLURM_NNODES" -gt 1 ]; then
    echo "ERROR: this script only works for a single node."
    exit 1
fi

if [ "$SLURM_NTASKS" -gt 1 ]; then
    echo "ERROR: this script needs to be run as a single task."
    exit 1
fi

if [ ! -z $NUM_GPUS ]; then
    SCRIPT_OPTS="--gradient_accumulation_steps $(( 8 / $NUM_GPUS ))"
fi

(set -x
 srun python $SCRIPT \
      --model_name_or_path EleutherAI/gpt-neo-1.3B \
      --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
      --per_device_train_batch_size 2 --do_train \
      --output_dir $OUTPUT_DIR --overwrite_output_dir \
      --bf16 \
      --num_train_epochs 1 --dataloader_num_workers 7 $SCRIPT_OPTS $*
 )
 
rm -rf $OUTPUT_DIR
