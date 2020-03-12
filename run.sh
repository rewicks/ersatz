#!/bin/bash

#$ -l gpu=4,h_rt=124:0:0
#$ -q gpu.q@@1080
#$ -o /home/hltcoe/rwicks/ersatz/ET-parallel/qsub.lor
#$ -e /home/hltcoe/rwicks/ersatz/ET-parallel/qsub.error
#$ -m eab
#$ -M rwicks4@jhu.edu
#$ -N TEST

. ~/.bashrc

conda deactivate
conda activate py3

date
python --version

cd $ERSATZ'ET-parallel/'

pwd

#spm_model_path='europarl-9.spm8000.context.model'

window_size='4'
train_path=$ERSATZ'data/train/europarl-v9.spm8000.en'

#python $ERSATZ/ET-parallel/dataset.py $train_path --context-size=$window_size

#train_path=$ERSATZ'data/train/europarl-v9.spm8000.'$window_size'-context.en'

#python $ERSATZ/ET-parallel/trainer.py $train_path \
#    --batch_size=5000 \
#    --epochs=15 \
#    --output=models/europarl-v9.model \
#    --lr=5.0 \
#    --embed_size=256 \
#    --nhead=8 \
#    --nlayers=8 \


for test_file in $ERSATZ/data/test/wsj/*; do
    name=$(echo "$test_file" | tr "/" "\n" | tail -1)
    python $ERSATZ/ET-parallel/split.py europarl-v9.spm8000.context.model \
        --input="$test_file" \
        --output=proc/"$name"
done
