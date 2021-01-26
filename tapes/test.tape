global {
	in=(Scenario: en="/exp/rwicks/ersatz/data/raw/en/devtest/wmt/wmt20.en-cs.en" \
							"/exp/rwicks/ersatz/data/raw/en/devtest/wmt/wmt20.en-cs.en.copy")
	determiner_type=(Scenario: en="en")
}

task split : ersatz
	< model_path=$out/checkpoint.best@train
	> out
	:: input_file=@
	:: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=1
    :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuActionFlags
	{
	mkdir -p out
	for file in in;
	do
		OUTPATH=$(echo $file | rev | cut -d'/' -f1 | rev)
		cat $input_file | tr '\n' ' ' | PYTHONPATH=$ersatz python $ersatz/split.py $model_path > out/$OUTPATH
	done;
	}

task score : ersatz
	< in=@
	< in_dir=$out@split
	< log=$out/job.out@train
	> out
	:: embed_size=@
	:: left_size=@
	:: right_size=@
	:: transformer_nlayers=@
	{
	mkdir -p out
	for file in $in;
	do
		if [[ $transofmrer_nlayers != 0 ]];
		then
		ARCH="transformer"
		else
		ARCH="linear"
		fi
		OUTPATH=$(echo $file | rev | cut -d'/' -f1 | rev)
		PARAMS=$(grep "Training with" $log | rev | cut -d' ' -f1 | rev)
		PYTHONPATH=$ersatz python $ersatz/score.py $file $in_dir/$OUTPATH > $out/$OUTPATH
		FSCORE=$(grep "F1" $out/$OUTPATH)
		echo -e $PARAMS"\t"$embed_size"\t"$ARCH"\t"$left_size"\t"$right_size"\t"$FSCORE
	done;

	}
