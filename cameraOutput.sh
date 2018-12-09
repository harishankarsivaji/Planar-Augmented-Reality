#!/bin/sh 

#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH -A nuim01
#SBATCH -p DevQ

module load conda
source activate pytorch
python camera.py
