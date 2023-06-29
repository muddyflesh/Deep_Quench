#!/bin/bash

#SBATCH --nodes=32
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --job-name=Ising_2d
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rc4571@nyu.edu
#SBATCH --output=Dim_2.out

module purge
module load python/intel/3.8.6
module load openmpi/intel/4.0.5

source /home/rc4571/test_venv/bin/activate
mpirun /home/rc4571/test_venv/bin/python ./Dim_2.py
