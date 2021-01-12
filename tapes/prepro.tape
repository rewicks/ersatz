global {
	train_data_path=(Scenario:
		en_16000="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm16000wm.en"
		en_8000="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm8000wm.en"
		en_4000="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm4000wm.en"
		en_2000="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm2000wm.en"
		en_1000="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm1000wm.en"
		en_500="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm500wm.en"
		en_250="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm250wm.en"
		en_125="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm125wm.en"
		char_en="/exp/rwicks/ersatz/data/processed/en/train/nc_wsj/merged.nc-wsj.spm10000wm.char.en"
	)
	valid_data_path=(Scenario:
		en_16000="/exp/rwicks/ersatz/data/processed/en/dev/nc_wsj/merged.nc-wsj.spm16000wm.en"
		en_8000="/exp/rwicks/ersatz/data/processed/en/dev/nc_wsj/merged.nc-wsj.spm8000wm.en"
		en_4000="/exp/rwicks/ersatz/data/processed/en/dev/nc_wsj/merged.nc-wsj.spm4000wm.en"
		en_2000="/exp/rwicks/ersatz/data/processed/en/dev/nc_wsj/merged.nc-wsj.spm2000wm.en"
		en_1000="/exp/rwicks/ersatz/data/processed/en/dev/nc_wsj/merged.nc-wsj.spm1000wm.en"
		en_500="/exp/rwicks/ersatz/data/processed/en/dev/nc_wsj/merged.nc-wsj.spm500wm.en"
		en_250="/exp/rwicks/ersatz/data/processed/en/dev/nc_wsj/merged.nc-wsj.spm250wm.en"
		en_125="/exp/rwicks/ersatz/data/processed/en/dev/nc_wsj/merged.nc-wsj.spm125wm.en"
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


