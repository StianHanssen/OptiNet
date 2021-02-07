#!/bin/sh

#SBATCH --account=share-ie-idi
#SBATCH --partition=V100-IDI
#SBATCH --gres=gpu:V100:1
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
# Memory per node specification is in MB. It is optional. 
# The default limit is 3000MB per core.
#SBATCH --mem=15000
#SBATCH --job-name="stianrh_AMD_Training"
#SBATCH --output=terminal_output/train_log.out
#SBATCH --mail-user=stianrh@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME" 
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "We are using $SLURM_GPUS_ON_NODE GPUs per node"
echo "Total of $SLURM_NTASKS cores"

echo "CUDA_VISILE DEVICES: $CUDA_VISIBLE_DEVICES"
module purge
module load GCC/7.3.0-2.30
module load icc/2018.3.222-GCC-7.3.0-2.30
module load intel/2018b
module load OpenMPI/3.1.1
module load impi/2018.3.222
module load Python/3.6.6
module load CUDA/9.0.176
nvidia-smi

cd ..
source .env/neural_nets/bin/activate
python main.py
deactivate
uname -a