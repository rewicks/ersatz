task train : ersatz
	< train_path=$out_shuffle@make_train_data
	< valid_path=$out_shuffle@make_valid_data
	> out
	:: left_size=@
	:: right_size=@
	:: vocabulary_path=@
	:: batch_size=@
	:: min_epochs=@
	:: max_epochs=@
	:: transformer_nlayers=@
    :: linear_nlayers=@
	:: lr=@
	:: dropout=@
	:: embed_size=@
	:: activation_type=@
	:: nhead=@
	:: log_interval=@
	:: validation_interval=@
	:: eos_weight=@
	{
	mkdir -p $out
	PYTHONPATH=$ersatz python $ersatz/trainer.py \
	--left_size=$left_size \
	--right_size=$right_size \
	--output_path=$out \
	--transformer_nlayers=$transformer_nlayers \
	--activation_type=$activation_type \
	--linear_nlayers=$linear_nlayers \
	--min-epochs=$min_epochs \
	--max-epochs=$max_epochs \
	--lr=$lr \
	--dropout=$dropout \
	--embed_size=$embed_size \
	--nhead=$nhead \
	--log_interval=$log_interval \
	--validation_interval=$validation_interval \
	--eos_weight=$eos_weight \
	$train_path \
	$valid_path
}

