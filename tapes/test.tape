global {
	in=(Scenario: en="/exp/rwicks/ersatz/data/raw/en/devtest/wmt/wmt20.en-cs.en"
                    en_wp="/exp/rwicks/ersatz/data/raw/en/devtest/wmt/wmt20.en-cs.en"
                    en_shuffled="/exp/rwicks/ersatz/data/raw/en/devtest/wmt/wmt20.en-cs.en"
                    multilingual="")
	determiner_type=(Scenario: en="en"
                                en_wp="en"
                                en_shuffled="en"
                                multilingual="")
}

task split : ersatz
	< model_path=$out@train
	> out
	:: in=@
    :: determiner_type=@
	:: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=1
    :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuActionFlags
	{
	mkdir -p out
	for file in ${in[@]};
	do
		OUTPATH=$(echo $file | rev | cut -d'/' -f1 | rev)
		cat $file | tr '\n' ' ' | PYTHONPATH=$ersatz python $ersatz/split.py $model_path/checkpoint.best --determiner_type=$determiner_type > out/$OUTPATH
	done;
	}

task score : ersatz
	< in_dir=$out@split
	< log=$out@train
	> out
	:: in=@
    :: embed_size=@
	:: left_size=@
	:: right_size=@
	:: transformer_nlayers=@
    :: determiner_type=@
	:: vocab_path=@
    {
	mkdir -p out
	for file in ${in[@]};
	do
		if [[ $transformer_nlayers != 0 ]];
		then
		ARCH="transformer"
		else
		ARCH="linear"
		fi
		OUTPATH=$(echo $file | rev | cut -d'/' -f1 | rev)
		LOG_PATH=$(echo $log | rev | cut -d'/' -f2- | rev)/job.out
        PARAMS=$(grep "Training with" $LOG_PATH | rev | cut -d' ' -f1 | rev)
		PYTHONPATH=$ersatz python $ersatz/score.py $file $in_dir/$OUTPATH --determiner_type=$determiner_type > $out/$OUTPATH
        VOCAB_SIZE=$(echo $vocab_path | cut -d'.' -f2)
		FSCORE=$(grep "F1" $out/$OUTPATH | cut -d' ' -f2-)
		echo -e $PARAMS"\t"$embed_size"\t"$VOCAB_SIZE"\t"$ARCH"\t"$left_size"\t"$right_size"\t"$FSCORE
	done;

	}

