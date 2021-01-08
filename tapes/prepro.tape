global {
	train_data_path=(Scenario:
		en="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm10000wm.en"
		char_en="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm10000wm.char.en"
	)
	valid_data_path=(Scenario:
		en="/exp/rwicks/ersatz/data/processed/en/dev/nc_wsj/merged.nc-wsj.spm10000wm.en"
		char_en="/exp/rwicks/ersatz/data/processed/en/dev/nc_wsj/merged.nc-wsj.spm10000wm.char.en"
	)
}

task make_train_data : ersatz
	:: left_size=@ right_size=@ train_data_path=@
	> out_og out_shuffle
	{
	PYTHONPATH=. python $ersatz/dataset.py \
		--left-size $left_size \
		--right-size $right_size \
		--output_path out_og \
		$train_data_path

	shuf --random-source=<(get_seeded_random 14) out_og > out_shuffle
}

task make_valid_data : ersatz
	:: left_size=@ right_size=@ valid_data_path=@
	> out_og out_shuffle
	{
	PYTHONPATH=. python $ersatz/dataset.py \
		--left-size $left_size \
		--right-size $right_size \
		--output_path out_og \
		$valid_data_path
	shuf --random-source=<(get_seeded_random 14) out_og > out_shuffle
}


