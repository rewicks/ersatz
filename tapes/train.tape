task train : ersatz
	< train_path=$out_shuffle@make_train_data
	< valid_path=$out_shuffle@make_valid_data
	< vocab_path=$model@train_vocab
    > out
	:: left_size=@
	:: right_size=@
    :: languages=@
	:: batch_size=@
	:: min_epochs=@
	:: max_epochs=@
	:: transformer_nlayers=@
    :: linear_nlayers=@
	:: lr=@
	:: dropout=@
	:: embed_size=@
    :: factor_embed_size=@
	:: activation_type=@
	:: nhead=@
	:: log_interval=@
	:: validation_interval=@
	:: log_dir=@
    :: early_stopping=@
    :: eos_weight=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=1
    :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuActionFlags	
    {
        mkdir -p $out
        LOGEXT=$(echo $out | rev | cut -d'/' -f2 | rev)
        LOGDIR=$log_dir"/"$LOGEXT
        rm -rf $LOGDIR

        if [$factor_embed_size = "0"]; then
            FACT_VALUE="--source_factors"
            echo "$FACT_VALUE"
        else
            FACT_VALUE=""
        fi

        #PYTHONPATH=$ersatz python $ersatz/trainer.py \
        ersatz_train \
        --sentencepiece_path=$vocab_path \
        --left_size=$left_size \
        --right_size=$right_size \
        --output_path=$out \
        --transformer_nlayers=$transformer_nlayers \
        --activation_type=$activation_type \
        --linear_nlayers=$linear_nlayers \
        --min-epochs=$min_epochs \
        --max-epochs=$max_epochs \
        --lr=$lr \
        --batch_size=$batch_size \
        --dropout=$dropout \
        --embed_size=$embed_size \
        --factor_embed_size=$factor_embed_size $FACT_VALUE \
        --nhead=$nhead \
        --log_interval=$log_interval \
        --validation_interval=$validation_interval \
        --eos_weight=$eos_weight \
        --early_stopping=$early_stopping \
        --tb_dir=$LOGDIR \
        --train_path=$train_path \
        --valid_path=$valid_path
}

