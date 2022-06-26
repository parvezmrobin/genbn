#!/bin/bash

# https://slurm.schedmd.com/sbatch.html

#SBATCH --output=logs/slurm-%x-%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=parvezmrobin@dal.ca

module load python/3.9
source venv/bin/activate
pip install -q -r requirements.txt

python tokenizer.py
