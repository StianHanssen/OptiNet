#!/bin/sh

#SBATCH --account=ie-idi
#SBATCH --partition=WORKQ
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Memory per node specification is in MB. It is optional. 
# The default limit is 3000MB per core.
#SBATCH --job-name="stianrh_AMD_Refiner"
#SBATCH --output=terminal_output/refiner_log.out

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
echo "Selected refiner id: $1"

module purge
module load GCC/7.3.0-2.30
module load icc/2018.3.222-GCC-7.3.0-2.30
module load intel/2018b
module load OpenMPI/3.1.1
module load impi/2018.3.222
module load Python/3.6.6

declare -A files=(["duke"]="mat_dataset_refiner.py" ["st_olavs"]="patient_dataset_refiner.py")

cd ..
source .env/neural_nets/bin/activate
cd dataset_refiners
python ${files[$1]}
deactivate
uname -a