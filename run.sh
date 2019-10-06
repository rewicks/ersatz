#!/bin/bash

#$ -l gpu=1,h_rt=12:00:00 -q gpu.q@@1080

. ~/.bashrc

conda deactivate
conda activate py3

python -u ~/awd-lstm-lm/main.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 --optimizer adam --lr 1e-3 --data europarl-v9 --save EUROPARL-V9-QRNN.pt --when 12 --model QRNN

