global {
    in_languages=(Scenario: en="en"
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
    baseline_type=(Base: punkt="punkt"                                                                                                                          
                            moses="moses"                                                                                                                       
                            ml_punkt="ml-punkt" always="always-split" never="never-split" spacy="spacy-split") 
}

task split : ersatz
	< model_path=$out@train
	> out
	:: in_languages=@
	:: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=1
    :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuActionFlags
	{
	mkdir -p $out
	for lang in ${in_languages[@]};
	do
        FILE_PATH="/exp/rwicks/ersatz/data/concat/test/$lang"
		OUTPATH=$(echo $FILE_PATH | rev | cut -d'/' -f1 | rev)
		cat $FILE_PATH | ~mpost/bin/strip_punc_rachel.py | tr '\n' ' ' | PYTHONPATH=$ersatz python $ersatz/split.py $model_path/checkpoint.best --determiner_type=multilingual > out/$OUTPATH
	done;
	}

task split_dev : ersatz
	< model_path=$out@train
	> out
	:: in_languages=@
	:: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=1
    :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuActionFlags
	{
	mkdir -p $out
	for lang in ${in_languages[@]};
	do
        FILE_PATH="/exp/rwicks/ersatz/data/concat/dev/$lang"
		OUTPATH=$(echo $FILE_PATH | rev | cut -d'/' -f1 | rev)
		cat $FILE_PATH | ~mpost/bin/strip_punc_rachel.py | tr '\n' ' ' | PYTHONPATH=$ersatz python $ersatz/split.py $model_path/checkpoint.best --determiner_type=multilingual > out/$OUTPATH
	done;
	}

task split_baseline : ersatz
    > out
    :: in_languages=@
    :: baseline_type=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
    {
    mkdir -p $out
    echo $baseline_type
    for lang in ${in_languages[@]};
    do
        FILE_PATH="/exp/rwicks/ersatz/data/concat/test/$lang"
        cat $FILE_PATH | ~mpost/bin/strip_punc_rachel.py | tr '\n' ' ' | $baseline_type --lang $lang > out/$lang
    done;
    }


task score : ersatz
	< in_dir=$out@split
	< log=$out@train
	> out
	:: in_languages=@
    :: embed_size=@
	:: left_size=@
	:: right_size=@
	:: transformer_nlayers=@
	:: vocab_size=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
    {
	mkdir -p $out
	for lang in ${in_languages[@]};
	do
        FILE_PATH="/exp/rwicks/ersatz/data/concat/test/$lang"
		if [[ $transformer_nlayers != 0 ]];
		then
		ARCH="transformer"
		else
		ARCH="linear"
		fi
		LOG_PATH=$(echo $log | rev | cut -d'/' -f2- | rev)/job.out
        PARAMS=$(grep "Training with" $LOG_PATH | rev | cut -d' ' -f1 | rev)
        cat $FILE_PATH | ~mpost/bin/strip_punc_rachel.py > new_gold.txt
		PYTHONPATH=$ersatz python $ersatz/score.py new_gold.txt $in_dir/$lang --determiner_type=multilingual > $out/$lang
		FSCORE=$(grep "F1" $out/$lang | tail -1 | cut -d' ' -f2-)
        echo -e $lang"\t"$PARAMS"\t"$embed_size"\t"$vocab_size"\t"$ARCH"\t"$left_size"\t"$right_size"\t"$FSCORE
	done;

	}

task score_dev : ersatz
	< in_dir=$out@split
	< log=$out@train
	> out
	:: in_languages=@
    :: embed_size=@
	:: left_size=@
	:: right_size=@
	:: transformer_nlayers=@
	:: vocab_size=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
    {
	mkdir -p $out
	for lang in ${in_languages[@]};
	do
        FILE_PATH="/exp/rwicks/ersatz/data/concat/dev/$lang"
		if [[ $transformer_nlayers != 0 ]];
		then
		ARCH="transformer"
		else
		ARCH="linear"
		fi
		LOG_PATH=$(echo $log | rev | cut -d'/' -f2- | rev)/job.out
        PARAMS=$(grep "Training with" $LOG_PATH | rev | cut -d' ' -f1 | rev)
        cat $FILE_PATH | ~mpost/bin/strip_punc_rachel.py > new_gold.txt
		PYTHONPATH=$ersatz python $ersatz/score.py new_gold.txt $in_dir/$lang --determiner_type=multilingual > $out/$lang
		FSCORE=$(grep "F1" $out/$lang | tail -1 | cut -d' ' -f2-)
        echo -e $lang"\t"$PARAMS"\t"$embed_size"\t"$vocab_size"\t"$ARCH"\t"$left_size"\t"$right_size"\t"$FSCORE
	done;

	}

task score_baseline : ersatz
    < in_dir=$out@split_baseline
    > out
    :: in_languages=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
    {
    mkdir -p $out
    for lang in ${in_languages[@]};
    do
        FILE_PATH="/exp/rwicks/ersatz/data/concat/test/$lang"
        cat $FILE_PATH | ~mpost/bin/strip_punc_rachel.py > new_gold.txt
        PYTHONPATH=$ersatz python $ersatz/score.py new_gold.txt $in_dir/$lang --determiner_type=multilingual > $out/$lang
        FSCORE=$(grep "F1" $out/$lang | tail -1 |cut -d' ' -f2-)
        RECALL=$(grep "Recall" $out/$lang | tail -1 | cut -d' ' -f2-)
        PRECISION=$(grep "Prec" $out/$lang | tail -1 | cut -d' ' -f2-)
        ACCURACY=$(grep "Acc" $out/$lang | tail -1 | cut -d' ' -f2-)
        echo -e $lang"\t"$FSCORE"\t"$RECALL"\t"$PRECISION"\t"$ACCURACY
    done;
    }
