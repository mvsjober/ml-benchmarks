#!/bin/bash

SBATCH="sbatch --parsable"
SBATCH_TEST="$SBATCH --account=project_2001659 --partition=test -t 5"

if [[ $HOSTNAME == *mahti.csc.fi ]]; then
    CLUSTER="mahti"
    GPUSMALL="gpusmall"
    GPUMEDIUM="gpumedium"
elif [[ $HOSTNAME == puhti-login* ]]; then
    CLUSTER="puhti"
    GPUSMALL="gpu"
    GPUMEDIUM="gpu"
elif [[ $HOSTNAME == uan* ]]; then
    CLUSTER="lumi"
else
    echo "ERROR: cannot determine cluster from hostname: $HOSTNAME"
    exit 1
fi

echo "Detected $CLUSTER cluster"

if [ "$LMOD_FAMILY_PYTHON_ML_ENV" != "pytorch" ]
then
    echo "WARNING: no pytorch module loaded, loading default module"
    module load pytorch
fi

JIDS=""

do_sbatch () {
    JID=$($SBATCH $*)
    echo "Submitted job $JID: $*"
    JIDS="$JIDS:$JID"
}

PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
echo "PyTorch version $PYTORCH_VERSION"

#### PyTorch DDP - syntethic data

# PyTorch DDP, single GPU
do_sbatch slurm/${CLUSTER}-gpu1.sh pytorch-ddp.sh --steps=1000
JID_DDP_GPU1=$JID

# PyTorch DDP, 4 GPU
do_sbatch --partition=$GPUMEDIUM -t 30 slurm/${CLUSTER}-gpu4.sh pytorch-ddp.sh
JID_DDP_GPU4=$JID

# PyTorch DDP multi-node, 8 GPU
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu8.sh pytorch-ddp.sh
JID_DDP_GPU8=$JID

#### PyTorch DDP - real data

# PyTorch DDP, single GPU, data
do_sbatch --partition=$GPUSMALL slurm/${CLUSTER}-gpu1.sh pytorch-ddp.sh --data --steps=1000
JID_DDP_DATA_GPU1=$JID

# PyTorch DDP, 4 GPU, data
do_sbatch --partition=$GPUMEDIUM -t 30 slurm/${CLUSTER}-gpu4.sh pytorch-ddp.sh --data
JID_DDP_DATA_GPU4=$JID

# PyTorch DDP multi-node, 8 GPU, data
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu8.sh pytorch-ddp.sh --data
JID_DDP_DATA_GPU8=$JID

#### PyTorch DeepSpeed

# PyTorch DeepSpeed, 4 GPU
do_sbatch --partition=$GPUMEDIUM -t 30 slurm/${CLUSTER}-gpu4.sh pytorch-deepspeed.sh
JID_DEEPSPEED_GPU4=$JID

# PyTorch DeepSpeed multi-node 8 GPU
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu8-mpi.sh pytorch-deepspeed.sh
JID_DEEPSPEED_GPU8=$JID

#### PyTorch Horovod

# PyTorch Horovod multi-node, 8 GPU with MPI
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu8-mpi.sh pytorch-horovod.sh
JID_HVD_GPU8=$JID

# PyTorch Horovod multi-node, 8 GPU with MPI
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu8-mpi.sh pytorch-horovod.sh --data
JID_HVD_DATA_GPU8=$JID

#### Summary
JID_SUMMARY=$($SBATCH_TEST --dependency=afterany$JIDS --job-name="results" --output="%x-%j.out" <<EOF
#!/bin/bash

print_result () {
    DESC=\$1
    NGPU=\$2
    JID=\$3
    echo -n "\$| DESC | PyTorch $PYTORCH_VERSION | $CLUSTER    | \$NGPU    | "
    LOGFN=\$(ls -1 logs/slurm-*-\$JID.out)
    RES=\$(grep '^Images/sec' \$LOGFN | tail -n1 | cut -d ' ' -f 2)
    if [ -z "\$RES" ]; then
       echo "ERROR IN \$LOGFN"
    else
       echo "\$RES |
    fi
}

print_result "PyTorch DDP, GPU1, synthetic" $JID_DDP_GPU1
print_result "PyTorch DDP, GPU4, synthetic" $JID_DDP_GPU4
print_result "PyTorch DDP, GPU8, synthetic" $JID_DDP_GPU8

print_result "PyTorch DDP, GPU1, Imagenet data" $JID_DDP_DATA_GPU1
print_result "PyTorch DDP, GPU4, Imagenet data" $JID_DDP_DATA_GPU4
print_result "PyTorch DDP, GPU8, Imagenet data" $JID_DDP_DATA_GPU8

print_result "DeepSpeed, synthetic data" 4 $JID_DEEPSPEED_GPU4
print_result "DeepSpeed, synthetic data" 8 $JID_DEEPSPEED_GPU8

print_result "PyTorch Horovod, GPU8, synthetic" $JID_HVD_GPU8
print_result "PyTorch Horovod, GPU8, Imagenet data" $JID_HVD_DATA_GPU8

EOF
)

# squeue -j "$JID_SUMMARY${JIDS//:/,}"
echo
echo "Submitted jobs: $JID_SUMMARY ${JIDS//:/ }"

echo
echo "Final summary will appear in results-${JID_SUMMARY}.out"
