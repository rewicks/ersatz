global {
        languages=(Scenario: en="en"
                             ar="ar"
                             cs="cs"
                             de="de"
                             es="es"
                             et="et"
                             fi="fi"
                             fr="fr"
                             gu="gu"
                             hi="hi"
                             iu="iu"
                             ja="ja"
                             kk="kk"
                             km="km"
                             lt="lt"
                             lv="lv"
                             pl="pl"
                             ps="ps"
                             ro="ro"
                             ru="ru"
                             ta="ta"
                             tr="tr"
                             zh="zh"
                             multilingual="ar cs de en es et fi fr gu hi iu ja kk km lt lv pl ps ro ru ta tr zh")   
}

task train_vocab : ersatz
    :: vocab_size=@
    :: languages=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
    > model vocab
    {
        mkdir -p out
        TRAIN_DATA_PATH=""
        for lang in $languages; do
            TRAIN_DATA_PATH=/exp/rwicks/ersatz/data/concat/train/$lang","$TRAIN_DATA_PATH
        done;
        TRAIN_DATA_PATH=${TRAIN_DATA_PATH%?}
        spm_train_py --input $TRAIN_DATA_PATH \
            --input_sentence_size 10000000 \
            --model_prefix out/ersatz \
            --vocab_size $vocab_size \
            --bos_piece "<mos>" \
            --eos_piece "<eos>" 
        ln -s `realpath out/ersatz.model` $model
        ln -s `realpath out/ersatz.vocab` $vocab
    }

task make_train_data : ersatz
    < vocab_path=$model@train_vocab
    :: left_size=@ right_size=@ languages=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
    > out_og out_shuffle
    {
    if [[ $languages =~ \ |\' ]]; then
        TRAIN_DATA_PATH=""
        for lang in $languages; do
            TRAIN_DATA_PATH="${TRAIN_DATA_PATH} /exp/rwicks/ersatz/data/concat/train/$lang"
        done
    else
        TRAIN_DATA_PATH="/exp/rwicks/ersatz/data/concat/train/$languages"
    fi
    PYTHONPATH=. python $ersatz/dataset.py \
        --sentencepiece_path $vocab_path \
        --left-size $left_size \
        --right-size $right_size \
        --output_path out_og \
        $TRAIN_DATA_PATH

    shuf --random-source=<(get_seeded_random 14) out_og > out_shuffle
}

task make_valid_data : ersatz
    < vocab_path=$model@train_vocab
    :: left_size=@ right_size=@ languages=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
    > out_og out_shuffle
    {
    if [[ $languages =~ \ |\' ]]; then
        VALID_DATA_PATH=""
        for lang in $languages; do
            VALID_DATA_PATH="${VALID_DATA_PATH} /exp/rwicks/ersatz/data/concat/dev/$lang"
        done
    else
        VALID_DATA_PATH="/exp/rwicks/ersatz/data/concat/dev/$languages"
    fi
    PYTHONPATH=. python $ersatz/dataset.py \
        --sentencepiece_path $vocab_path \
        --left-size $left_size \
        --right-size $right_size \
        --output_path out_og \
        $VALID_DATA_PATH
    shuf --random-source=<(get_seeded_random 14) out_og > out_shuffle
}

