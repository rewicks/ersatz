global {
	in=(Scenario: en="/exp/rwicks/ersatz/data/raw/en/devtest/wmt/wmt20.en-cs.en"
                    en_wp="/exp/rwicks/ersatz/data/raw/en/devtest/wmt/wmt20.en-cs.en"
                    en_shuffled="/exp/rwicks/ersatz/data/raw/en/devtest/wmt/wmt20.en-cs.en"
                    multilingual="/exp/rwicks/ersatz/data/raw/ar/test/iwslt/iwsltt2017.ar /exp/rwicks/ersatz/data/raw/cs/test/wmt/wmt20.cs-en.cs /exp/rwicks/ersatz/data/raw/de/test/wmt/wmt20.de-en.de /exp/rwicks/ersatz/data/raw/en/test/wsj/wsj.03-06.en /exp/rwicks/ersatz/data/raw/es/test/wmt/wmt13.es-en.es /exp/rwicks/ersatz/data/raw/et/test/wmt/wmt18.et-en.et /exp/rwicks/ersatz/data/raw/fi/test/wmt/wmt19.fi-en.fi /exp/rwicks/ersatz/data/raw/fr/test/wmt/wmt20.fr-de.fr /exp/rwicks/ersatz/data/raw/gu/test/wmt/wmt19.gu-en.gu /exp/rwicks/ersatz/data/raw/hi/test/wmt/wmt14.hi-en.hi /exp/rwicks/ersatz/data/raw/iu/test/wmt/wmt20.iu-en.iu /exp/rwicks/ersatz/data/raw/ja/test/wmt/wmt20.ja-en.ja /exp/rwicks/ersatz/data/raw/kk/test/wmt/wmt19.kk-en.kk /exp/rwicks/ersatz/data/raw/km/test/wmt/wmt20.km-en.km /exp/rwicks/ersatz/data/raw/lv/test/wmt/wmt17.lv-en.lv /exp/rwicks/ersatz/data/raw/lt/test/wmt/wmt19.lt-en.lt /exp/rwicks/ersatz/data/raw/pl/test/wmt/wmt20.pl-en.pl /exp/rwicks/ersatz/data/raw/ps/test/wmt/wmt20.ps-en.ps /exp/rwicks/ersatz/data/raw/ro/test/wmt/wmt16.ro-en.ro /exp/rwicks/ersatz/data/raw/ru/test/wmt/wmt20.ru-en.ru /exp/rwicks/ersatz/data/raw/ta/test/wmt/wmt20.ta-en.ta /exp/rwicks/ersatz/data/raw/tr/test/wmt/wmt18.tr-en.tr /exp/rwicks/ersatz/data/raw/zh/test/wmt/wmt20.zh-en.zh"
                    multilingual_large="/exp/rwicks/ersatz/data/raw/ar/test/iwslt/iwsltt2017.ar /exp/rwicks/ersatz/data/raw/cs/test/wmt/wmt20.cs-en.cs /exp/rwicks/ersatz/data/raw/de/test/wmt/wmt20.de-en.de /exp/rwicks/ersatz/data/raw/en/test/wsj/wsj.03-06.en /exp/rwicks/ersatz/data/raw/es/test/wmt/wmt13.es-en.es /exp/rwicks/ersatz/data/raw/et/test/wmt/wmt18.et-en.et /exp/rwicks/ersatz/data/raw/fi/test/wmt/wmt19.fi-en.fi /exp/rwicks/ersatz/data/raw/fr/test/wmt/wmt20.fr-de.fr /exp/rwicks/ersatz/data/raw/gu/test/wmt/wmt19.gu-en.gu /exp/rwicks/ersatz/data/raw/hi/test/wmt/wmt14.hi-en.hi /exp/rwicks/ersatz/data/raw/iu/test/wmt/wmt20.iu-en.iu /exp/rwicks/ersatz/data/raw/ja/test/wmt/wmt20.ja-en.ja /exp/rwicks/ersatz/data/raw/kk/test/wmt/wmt19.kk-en.kk /exp/rwicks/ersatz/data/raw/km/test/wmt/wmt20.km-en.km /exp/rwicks/ersatz/data/raw/lv/test/wmt/wmt17.lv-en.lv /exp/rwicks/ersatz/data/raw/lt/test/wmt/wmt19.lt-en.lt /exp/rwicks/ersatz/data/raw/pl/test/wmt/wmt20.pl-en.pl /exp/rwicks/ersatz/data/raw/ps/test/wmt/wmt20.ps-en.ps /exp/rwicks/ersatz/data/raw/ro/test/wmt/wmt16.ro-en.ro /exp/rwicks/ersatz/data/raw/ru/test/wmt/wmt20.ru-en.ru /exp/rwicks/ersatz/data/raw/ta/test/wmt/wmt20.ta-en.ta /exp/rwicks/ersatz/data/raw/tr/test/wmt/wmt18.tr-en.tr /exp/rwicks/ersatz/data/raw/zh/test/wmt/wmt20.zh-en.zh")
	determiner_type=(Scenario: en="en"
                                en_wp="en"
                                en_shuffled="en"
                                multilingual="multilingual"
                                multilingual_large="multilingual")
    baseline_type=(Base: punkt="punkt"                                                                                                                          
                            moses="moses"                                                                                                                       
                            ml_punkt="ml-punkt" always="always-split" never="never-split" spacy="spacy-split") 
}

task split : ersatz
	< model_path=$out@train
	> out
	:: in=@
    :: determiner_type=@
	#:: pyenv=@ :: .submitter=$grid :: devices=@
    #:: devices_per_task=1
    #:: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuActionFlags
	{
	mkdir -p $out
	for file in ${in[@]};
	do
		OUTPATH=$(echo $file | rev | cut -d'/' -f1 | rev)
		cat $file | ~mpost/bin/strip_punc_rachel.py | tr '\n' ' ' | PYTHONPATH=$ersatz python $ersatz/split.py $model_path/checkpoint.best --determiner_type=$determiner_type > out/$OUTPATH
	done;
	}

task split_baseline : ersatz
    > out
    :: in=@
    :: baseline_type=@
    {
    mkdir -p $out
    echo $baseline_type
    for file in ${in[@]};
    do
        OUTPATH=$(echo $file | rev | cut -d'/' -f1 | rev)
        LANG=$(echo $file | rev | cut -d'.' -f1 | rev)
        cat $file | ~mpost/bin/strip_punc_rachel.py | tr '\n' ' ' | $baseline_type --lang $LANG > out/$OUTPATH
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
	mkdir -p $out
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
        LANG=$(echo $file | rev | cut -d'.' -f1 | rev)
        PARAMS=$(grep "Training with" $LOG_PATH | rev | cut -d' ' -f1 | rev)
        cat $file | ~mpost/bin/strip_punc_rachel.py > new_gold.txt
		PYTHONPATH=$ersatz python $ersatz/score.py new_gold.txt $in_dir/$OUTPATH --determiner_type=$determiner_type > $out/$OUTPATH
        VOCAB_SIZE=$(echo $vocab_path | cut -d'.' -f2)
		FSCORE=$(grep "F1" $out/$OUTPATH | tail -1 | cut -d' ' -f2-)
        echo -e $LANG"\t"$PARAMS"\t"$embed_size"\t"$VOCAB_SIZE"\t"$ARCH"\t"$left_size"\t"$right_size"\t"$FSCORE
	done;

	}

task score_baseline : ersatz
    < in_dir=$out@split_baseline
    > out
    :: in=@
    :: determiner_type=@
    {
    mkdir -p $out
    for file in ${in[@]};
    do
        OUTPATH=$(echo $file | rev | cut -d'/' -f1 | rev)
        LANG=$(echo $file | rev | cut -d'.' -f1 | rev)
        cat $file | ~mpost/bin/strip_punc_rachel.py > new_gold.txt
        PYTHONPATH=$ersatz python $ersatz/score.py new_gold.txt $in_dir/$OUTPATH --determiner_type=$determiner_type > $out/$OUTPATH
        FSCORE=$(grep "F1" $out/$OUTPATH | tail -1 |cut -d' ' -f2-)
        RECALL=$(grep "Recall" $out/$OUTPATH | tail -1 | cut -d' ' -f2-)
        PRECISION=$(grep "Prec" $out/$OUTPATH | tail -1 | cut -d' ' -f2-)
        ACCURACY=$(grep "Acc" $out/$OUTPATH | tail -1 | cut -d' ' -f2-)
        echo -e $LANG"\t"$FSCORE"\t"$RECALL"\t"$PRECISION"\t"$ACCURACY
    done;
    }
