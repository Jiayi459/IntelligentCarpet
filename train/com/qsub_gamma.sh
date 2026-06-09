#!/bin/bash
#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -l h_rt=14:00:00
#$ -N gamma_fusion
#$ -o train/com/output/phase2_gamma/qsub.out
#$ -e train/com/output/phase2_gamma/qsub.err

module load conda
module load cuda/12.1
conda activate carpet
cd ~/IntelligentCarpet
python -u train/com/train_phase2_gamma.py
