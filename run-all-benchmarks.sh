#!/bin/bash

SBATCH="sbatch --parsable"
SBATCH_TEST="$SBATCH --account=project_2001659 --partition=test -t 5"

if [[ $HOSTNAME == *mahti.csc.fi ]]; then
    CLUSTER="mahti"
elif [[ $HOSTNAME == puhti-login* ]]; then
    CLUSTER="puhti"
elif [[ $HOSTNAME == uan* ]]; then
    CLUSTER="lumi"
else
    echo "ERROR: cannot determine cluster from hostname: $HOSTNAME"
    exit 1
fi

JIDS=""

do_sbatch () {
    JID=$($SBATCH $*)
    echo "Submitted job $JID: $*"
    JIDS="$JIDS:$JID"
}

#### PyTorch DDP - syntethic data

# PyTorch DDP, single GPU
do_sbatch --partition=gpusmall slurm/${CLUSTER}-gpu1.sh pytorch-ddp.sh --steps=1000
JID_DDP_GPU1=$JID

# PyTorch DDP, 4 GPU
do_sbatch --partition=gpumedium slurm/${CLUSTER}-gpu4.sh pytorch-ddp.sh
JID_DDP_GPU4=$JID

# PyTorch DDP multi-node, 8 GPU
do_sbatch slurm/${CLUSTER}-gpu8.sh pytorch-ddp.sh
JID_DDP_GPU8=$JID

#### PyTorch DDP - real data

# PyTorch DDP, single GPU, data
do_sbatch --partition=gpusmall slurm/${CLUSTER}-gpu1.sh pytorch-ddp.sh --data --steps=1000
JID_DDP_DATA_GPU1=$JID

# PyTorch DDP, 4 GPU, data
do_sbatch --partition=gpumedium slurm/${CLUSTER}-gpu4.sh pytorch-ddp.sh --data
JID_DDP_DATA_GPU4=$JID

# PyTorch DDP multi-node, 8 GPU, data
do_sbatch slurm/${CLUSTER}-gpu8.sh pytorch-ddp.sh --data
JID_DDP_DATA_GPU8=$JID

#### PyTorch Horovod

# PyTorch Horovod multi-node, 8 GPU with MPI
do_sbatch slurm/${CLUSTER}-gpu8-mpi.sh pytorch-horovod.sh
JID_HVD_GPU8=$JID

# PyTorch Horovod multi-node, 8 GPU with MPI
do_sbatch slurm/${CLUSTER}-gpu8-mpi.sh pytorch-horovod.sh --data
JID_HVD_DATA_GPU8=$JID

#### Summary
JID_SUMMARY=$($SBATCH_TEST --dependency=afterany$JIDS --job-name="benchmark-summary" <<EOF
#!/bin/bash

print_result () {
    DESC=\$1
    JID=\$2
    echo -n "\$DESC | "
    LOGFN=\$(ls -1 logs/slurm-*-\$JID.out)
    RES=\$(grep '^Images/sec' \$LOGFN | cut -d ' ' -f 2)
    if [ -z \$RES ]; then
       echo "ERROR IN \$LOGFN"
    else
       echo \$RES
    fi
}

print_result "PyTorch DDP, GPU1, synthetic" $JID_DDP_GPU1
print_result "PyTorch DDP, GPU4, synthetic" $JID_DDP_GPU4
print_result "PyTorch DDP, GPU8, synthetic" $JID_DDP_GPU8

print_result "PyTorch DDP, GPU1, Imagenet data" $JID_DDP_DATA_GPU1
print_result "PyTorch DDP, GPU4, Imagenet data" $JID_DDP_DATA_GPU4
print_result "PyTorch DDP, GPU8, Imagenet data" $JID_DDP_DATA_GPU8

print_result "PyTorch Horovod, GPU8, synthetic" $JID_HVD_GPU8
print_result "PyTorch Horovod, GPU8, Imagenet data" $JID_HVD_DATA_GPU8

EOF
)

# squeue -j "$JID_SUMMARY${JIDS//:/,}"
echo
echo "Submitted jobs: $JID_SUMMARY ${JIDS//:/ }"

echo
echo "Final summary will appear in slurm-${JID_SUMMARY}.out"
