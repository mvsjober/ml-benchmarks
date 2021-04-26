SCRIPT=$1
shift
cd $SLURM_SUBMIT_DIR

module list

set -x

date
hostname
nvidia-smi

source $SCRIPT $*

date
