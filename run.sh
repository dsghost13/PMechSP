#!/bin/bash
#SBATCH --job-name=10k-Mayr
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=petertl2@uci.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=7-24:00:00
#SBATCH --account=dlvanvra_lab

/data/homezvol0/petertl2/.conda/envs/chatnt/bin/python /data/homezvol0/petertl2/PMechSP/run_chatnt.py

