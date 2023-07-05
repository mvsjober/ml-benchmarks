#!/bin/bash

SBATCH="sbatch --parsable"
SBATCH_TEST="$SBATCH --account=project_2001659 --partition=test -t 5"

if [[ $HOSTNAME == *mahti.csc.fi ]]; then
    CLUSTER="mahti"
    GPUSMALL="gpusmall"
    GPUMEDIUM="gpumedium"
    FULLNODE="4"
    TWONODES="8"
elif [[ $HOSTNAME == puhti-login* ]]; then
    CLUSTER="puhti"
    GPUSMALL="gpu"
    GPUMEDIUM="gpu"
    FULLNODE="4"
    TWONODES="8"
elif [[ $HOSTNAME == uan* ]]; then
    CLUSTER="lumi"
    GPUSMALL="small-g"
    GPUMEDIUM="small-g"
    FULLNODE="8"
    TWONODES="16"
    SBATCH_TEST="$SBATCH --account=project_462000007 --partition=small -t 5"
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

# PyTorch DDP, full node
do_sbatch --partition=$GPUMEDIUM -t 30 slurm/${CLUSTER}-gpu${FULLNODE}.sh pytorch-ddp.sh
JID_DDP_FULLNODE=$JID

# PyTorch DDP multi-node, two nodes
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu${TWONODES}.sh pytorch-ddp.sh
JID_DDP_TWONODES=$JID


#### PyTorch DDP Lightning - syntethic data

# PyTorch DDP Lightning, single GPU
do_sbatch --partition=$GPUSMALL -t 30 slurm/${CLUSTER}-gpu1.sh pytorch-ddp-lightning.sh --steps=1000
JID_DDPL_GPU1=$JID

# PyTorch DDP, full node
do_sbatch --partition=$GPUMEDIUM -t 30 slurm/${CLUSTER}-gpu${FULLNODE}-mpi.sh pytorch-ddp-lightning.sh
JID_DDPL_FULLNODE=$JID

# PyTorch DDP multi-node, two nodes
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu${TWONODES}-mpi.sh pytorch-ddp-lightning.sh
JID_DDPL_TWONODES=$JID


if [ "$CLUSTER" != "lumi" ]; then
#### PyTorch DDP - real data

# PyTorch DDP, single GPU, data
do_sbatch --partition=$GPUSMALL slurm/${CLUSTER}-gpu1.sh pytorch-ddp.sh --data --steps=1000
JID_DDP_DATA_GPU1=$JID

# PyTorch DDP, 4 GPU, data
do_sbatch --partition=$GPUMEDIUM -t 30 slurm/${CLUSTER}-gpu${FULLNODE}.sh pytorch-ddp.sh --data
JID_DDP_DATA_FULLNODE=$JID

# PyTorch DDP multi-node, 8 GPU, data
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu${TWONODES}.sh pytorch-ddp.sh --data
JID_DDP_DATA_TWONODES=$JID

#### PyTorch DeepSpeed

# PyTorch DeepSpeed, 4 GPU
do_sbatch --partition=$GPUMEDIUM -t 30 slurm/${CLUSTER}-gpu${FULLNODE}.sh pytorch-deepspeed.sh
JID_DEEPSPEED_FULLNODE=$JID

# PyTorch DeepSpeed multi-node 8 GPU
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu${TWONODES}-mpi.sh pytorch-deepspeed.sh
JID_DEEPSPEED_TWONODES=$JID

#### PyTorch Horovod

# PyTorch Horovod multi-node, 8 GPU with MPI
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu${TWONODES}-mpi.sh pytorch-horovod.sh
JID_HVD_TWONODES=$JID

# PyTorch Horovod multi-node, 8 GPU with MPI
do_sbatch --partition=$GPUMEDIUM slurm/${CLUSTER}-gpu${TWONODES}-mpi.sh pytorch-horovod.sh --data
JID_HVD_DATA_TWONODES=$JID

fi

#### Summary

JID_SUMMARY=$($SBATCH_TEST --dependency=afterany$JIDS --job-name="results" --output="%x-%j.out" <<EOF
#!/bin/bash

print_result () {
    DESC=\$1
    NGPU=\$2
    JID=\$3
    DATENOW=\$(date +%F)
    echo -n "| \$DESC | PyTorch $PYTORCH_VERSION | $CLUSTER    | \$NGPU    | \$DATENOW | "
    LOGFN=\$(ls -1 logs/slurm-*-\$JID.out)
    RES=\$(grep '^Images/sec' \$LOGFN | tail -n1 | cut -d ' ' -f 2)
    if [ -z "\$RES" ]; then
       echo "ERROR IN \$LOGFN"
    else
       echo "\$RES |"
    fi
}

print_result "DDP, synthetic" 1 $JID_DDP_GPU1
print_result "DDP, synthetic" $FULLNODE $JID_DDP_FULLNODE
print_result "DDP, synthetic" ${TWONODES} $JID_DDP_TWONODES

print_result "DDP Lightning, synthetic" 1 $JID_DDPL_GPU1
print_result "DDP Lightning, synthetic" $FULLNODE $JID_DDPL_FULLNODE
print_result "DDP Lightning, synthetic" ${TWONODES} $JID_DDPL_TWONODES

print_result "DDP, Imagenet data" 1 $JID_DDP_DATA_GPU1
print_result "DDP, Imagenet data" $FULLNODE $JID_DDP_DATA_FULLNODE
print_result "DDP, Imagenet data" ${TWONODES} $JID_DDP_DATA_TWONODES

print_result "DeepSpeed, synthetic data" $FULLNODE $JID_DEEPSPEED_FULLNODE
print_result "DeepSpeed, synthetic data" ${TWONODES} $JID_DEEPSPEED_TWONODES

print_result "Horovod, synthetic" ${TWONODES} $JID_HVD_TWONODES
print_result "Horovod, Imagenet data" ${TWONODES} $JID_HVD_DATA_TWONODES

EOF
)

# squeue -j "$JID_SUMMARY${JIDS//:/,}"
echo
echo "Submitted jobs: $JID_SUMMARY ${JIDS//:/ }"

echo
echo "Final summary will appear in results-${JID_SUMMARY}.out"
