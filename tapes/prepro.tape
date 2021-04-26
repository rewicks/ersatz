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
task make_train_data : ersatz
	:: vocab_path=@ left_size=@ right_size=@ languages=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
    > out_og out_shuffle
	{
    TRAIN_DATA_PATH="/exp/rwicks/ersatz/data/concat/train/$languages"
    function has-space {
  	[[ "$1" != "${1%[[:space:]]*}" ]] && return 0 || return 1
	}
	if has-space $languages; then
		VOCAB_PATH=$vocab_path"multilingual.model"
	else
		VOCAB_PATH=$vocab_path$languages".model"
	fi
    PYTHONPATH=. python $ersatz/dataset.py \
		--sentencepiece_path $VOCAB_PATH \
        --left-size $left_size \
		--right-size $right_size \
		--output_path out_og \
		$TRAIN_DATA_PATH

	shuf --random-source=<(get_seeded_random 14) out_og > out_shuffle
}

task make_valid_data : ersatz
	:: vocab_path=@ left_size=@ right_size=@ languages=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
	> out_og out_shuffle
	{
    VALID_DATA_PATH="/exp/rwicks/ersatz/data/concat/dev/$languages"
    function has-space {
  	[[ "$1" != "${1%[[:space:]]*}" ]] && return 0 || return 1
	}
	if has-space $languages; then
		VOCAB_PATH=$vocab_path"multilingual.model"
	else
		VOCAB_PATH=$vocab_path$languages".model"
	fi
	PYTHONPATH=. python $ersatz/dataset.py \
        --sentencepiece_path $VOCAB_PATH \
		--left-size $left_size \
		--right-size $right_size \
		--output_path out_og \
		$VALID_DATA_PATH
	shuf --random-source=<(get_seeded_random 14) out_og > out_shuffle
}


