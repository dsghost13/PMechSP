#!/bin/bash
#SBATCH --job-name=10k-PT
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=petertl2@uci.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=7-24:00:00
#SBATCH --account=dlvanvra_lab

cd /data/homezvol0/petertl2/PMechSP


/data/homezvol0/petertl2/miniconda3/envs/PMechSP/bin/python /data/homezvol0/petertl2/PMechSP/scripts/train_PT.py