#!/bin/bash

#$ -l gpu=4,h_rt=124:0:0 -q gpu.q@@1080 -o /home/hltcoe/rwicks/ersatz/exp/11/qsub.log -e /home/hltcoe/rwicks/ersatz/exp/11/qsub.error -m e -M rwicks4@jhu.edu

. ~/.bashrc

conda deactivate
conda activate py3

date
python --version

cd $ERSATZ'ET-parallel/'

pwd

python trainer.py
