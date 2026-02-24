#!/bin/bash
#SBATCH --job-name=37k-Combined
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=petertl2@uci.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=7-24:00:00
#SBATCH --account=dlvanvra_lab

cd /data/homezvol0/petertl2/PMechSP/shells

/data/homezvol0/petertl2/miniconda3/envs/PMechSP/bin/python /data/homezvol0/petertl2/PMechSP/shells/train_combined.py
