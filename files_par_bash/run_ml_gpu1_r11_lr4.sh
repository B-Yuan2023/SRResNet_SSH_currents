#!/bin/bash
#SBATCH --job-name=out_interp      # Specify job name
#SBATCH --partition=gpu            # Specify partition name
#SBATCH --nodes=1                  # Specify number of nodes
#SBATCH --gpus=1                   # Specify number of GPUs needed for the job
#SBATCH --time=12:00:00            # Set a limit on the total run time
#SBATCH --mail-type=FAIL           # Notify user by email in case of job failure
#SBATCH --account=gg0028           # Charge resources on this project account
#SBATCH --output=out.o%j        # File name for standard output
#SBATCH --error=out.e%j        # File name for standard error output

module load python3/2022.01-gcc-11.2.0 
module load pytorch/1.13.0

# Run GPU-aware script, e.g.
set +k
cdir=$(pwd)
cd ${cdir}
python srren.py 'par11_lr4'
