#!/bin/bash
#SBATCH --account=def-mlefsrud         # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/<user>/slurm-%j.out  # Write the log in $SCRATCH# 1. Create your environement locally
module load python/3.7

virtualenv --no-download $SLURM_TMPDIR/plantrl

source $SLURM_TMPDIR/plantrl/bin/activate

pip install --no-index torch torchvision   # why is this not in virtualenv
pip install --no-index -r requirements_y.txt

# 2. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

python main.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR

# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/runtest1 $SCRATCH
