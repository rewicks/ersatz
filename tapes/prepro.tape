global {
	train_data_path=(Scenario: en="/exp/rwicks/ersatz/data/raw/en/train/nc_wsj/merged.nc-wsj.en")
	valid_data_path=(Scenario: en="/exp/rwicks/ersatz/data/raw/en/dev/nc_wsj/merged.nc-wsj.en")
}

task make_train_data : ersatz
	:: vocab_path=@ left_size=@ right_size=@ train_data_path=@
	> out_og out_shuffle
	{
	PYTHONPATH=. python $ersatz/dataset.py \
		--sentencepiece_path $vocab_path \
        --left-size $left_size \
		--right-size $right_size \
		--output_path out_og \
		$train_data_path

	shuf --random-source=<(get_seeded_random 14) out_og > out_shuffle
}

task make_valid_data : ersatz
	:: vocab_path=@ left_size=@ right_size=@ valid_data_path=@
	> out_og out_shuffle
	{
	PYTHONPATH=. python $ersatz/dataset.py \
        --sentencepiece_path $vocab_path \
		--left-size $left_size \
		--right-size $right_size \
		--output_path out_og \
		$valid_data_path
	shuf --random-source=<(get_seeded_random 14) out_og > out_shuffle
}


