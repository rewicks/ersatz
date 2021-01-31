global {
	train_data_path=(Scenario: en="/exp/rwicks/ersatz/data/raw/en/train/nc_wsj/merged.nc-wsj.en"
                                en_wp="/exp/rwicks/ersatz/data/raw/en/train/wiki-matrix/wiki.filtered.nostar.text.en"
                                en_shuffled="/exp/rwicks/ersatz/data/raw/en/train/nc_wsj/merged.nc-wsj.shuffled.en"
                                multilingual="/exp/rwicks/ersatz/data/raw/ar/train/news-commentary/news-commentary-v15.train.ar /exp/rwicks/ersatz/data/raw/cs/train/news-commentary/news-commentary-v15.cs /exp/rwicks/ersatz/data/raw/cs/train/wikipedia/wikimatrix.cs /exp/rwicks/ersatz/data/raw/de/train/news-commentary/news-commentary-v15.de /exp/rwicks/ersatz/data/raw/de/train/wikipedia/wikimatrix.de /exp/rwicks/ersatz/data/raw/en/train/news-commentary/news-commentary-v15.short.en /exp/rwicks/ersatz/data/raw/es/train/news-commentary/news-commentary-v15.short.es /exp/rwicks/ersatz/data/raw/et/train/news-crawl/news.2019.short.et /exp/rwicks/ersatz/data/raw/et/train/wikipedia/wikimatrix.et /exp/rwicks/ersatz/data/raw/fi/train/news-crawl/news.2019.short.fi /exp/rwicks/ersatz/data/raw/fi/train/wikipedia/wikimatrix.fi /exp/rwicks/ersatz/data/raw/fr/train/news-commentary/news-commentary-v15.fr /exp/rwicks/ersatz/data/raw/fr/train/wikipedia/wikimatrix.fr /exp/rwicks/ersatz/data/raw/gu/train/news-crawl/head.news.2019.gu.shuffled.deduped /exp/rwicks/ersatz/data/raw/gu/train/cc-100/gu.short.txt /exp/rwicks/ersatz/data/raw/hi/train/news-commentary/news-commentary-v15.hi /exp/rwicks/ersatz/data/raw/hi/train/wikipedia/wikimatrix.hi /exp/rwicks/ersatz/data/raw/hi/train/news-crawl/news.2013.short.hi /exp/rwicks/ersatz/data/raw/iu/train/Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0/train.short.iu /exp/rwicks/ersatz/data/raw/ja/train/news-commentary/news-commentary-v15.ja /exp/rwicks/ersatz/data/raw/ja/train/news-crawl/news.2019.short.ja /exp/rwicks/ersatz/data/raw/ja/train/wikipedia/wikimatrix.ja /exp/rwicks/ersatz/data/raw/kk/train/news-commentary/news-commentary-v15.kk /exp/rwicks/ersatz/data/raw/kk/train/news-crawl/news.2019.short.kk /exp/rwicks/ersatz/data/raw/kk/train/wikipedia/wikimatrix.kk /exp/rwicks/ersatz/data/raw/km/train/km-parallel/jw300-wmt20.en-km.km /exp/rwicks/ersatz/data/raw/km/train/cc-100/km.short.txt /exp/rwicks/ersatz/data/raw/lt/train/news-crawl/news.2019.short.lt /exp/rwicks/ersatz/data/raw/lt/train/wikipedia/wikimatrix.lt /exp/rwicks/ersatz/data/raw/lv/train/news-crawl/news.2019.short.lv /exp/rwicks/ersatz/data/raw/pl/train/global-voices/pl.txt /exp/rwicks/ersatz/data/raw/pl/train/news-crawl/news.2019.short.pl /exp/rwicks/ersatz/data/raw/pl/train/wikipedia/wikimatrix.pl /exp/rwicks/ersatz/data/raw/ps/train/news-crawl/news.2019.ps.shuffled.deduped /exp/rwicks/ersatz/data/raw/ps/train/sada/sada-v9.ps /exp/rwicks/ersatz/data/raw/ps/train/systran/systran-corpus-usg-201309.train.ps /exp/rwicks/ersatz/data/raw/ps/train/transtac/transtac.bilingual.bitext.clean.ps /exp/rwicks/ersatz/data/raw/ro/train/global-voices/ro.txt /exp/rwicks/ersatz/data/raw/ro/train/wikipedia/wikimatrix.ro /exp/rwicks/ersatz/data/raw/ro/train/news-crawl/news.2019.short.ro /exp/rwicks/ersatz/data/raw/ru/train/news-commentary/news-commentary-v15.ru /exp/rwicks/ersatz/data/raw/ru/train/wikipedia/wikimatrix.ru /exp/rwicks/ersatz/data/raw/ta/train/news-crawl/news.2019.short.ta /exp/rwicks/ersatz/data/raw/ta/train/wikipedia/wikimatrix.ta /exp/rwicks/ersatz/data/raw/tr/train/global-voices/tr.txt /exp/rwicks/ersatz/data/raw/tr/train/wikipedia/wikimatrix.tr /exp/rwicks/ersatz/data/raw/tr/train/news-crawl/news.2019.short.tr /exp/rwicks/ersatz/data/raw/zh/train/news-commentary/news-commentary-v15.zh /exp/rwicks/ersatz/data/raw/zh/train/wikipedia/wikimatrix.zh")
	valid_data_path=(Scenario: en="/exp/rwicks/ersatz/data/raw/en/dev/nc_wsj/merged.nc-wsj.en"
                                en_wp="/exp/rwicks/ersatz/data/raw/en/dev/nc_wsj/merged.nc-wsj.en"
                                en_shuffled="/exp/rwicks/ersatz/data/raw/en/dev/nc_wsj/merged.nc-wsj.en"
                                multilingual="/exp/rwicks/ersatz/data/raw/ar/dev/news-commentary/segmented.news-commentary-v15.dev.ar /exp/rwicks/ersatz/data/raw/cs/dev/wmt/wmt18.cs-en.cs /exp/rwicks/ersatz/data/raw/de/dev/wmt/wmt19.de-en.de /exp/rwicks/ersatz/data/raw/en/dev/nc_wsj/merged.nc-wsj.en /exp/rwicks/ersatz/data/raw/et/dev/news-commentary/segmented.tail.news.2019.et.shuffled.deduped /exp/rwicks/ersatz/data/raw/fi/dev/wmt/wmt18.fi-en.fi /exp/rwicks/ersatz/data/raw/fr/dev/wmt/wmt15.fr-en.fr /exp/rwicks/ersatz/data/raw/gu/dev/news-commentary/segmented.tail.news.2019.gu.shuffled.deduped /exp/rwicks/ersatz/data/raw/hi/dev/news-commentary/segmented.tail.news.2013.hi.shuffled /exp/rwicks/ersatz/data/raw/iu/dev/Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0/segmented.dev-dedup.iu /exp/rwicks/ersatz/data/raw/ja/dev/news-commentary/segmented.tail.news.2019.ja.shuffled.deduped /exp/rwicks/ersatz/data/raw/kk/dev/news-commentary/segmented.tail.news.2019.kk.shuffled.deduped /exp/rwicks/ersatz/data/raw/km/dev/wikipedia/segmented.wikipedia.dev.km-en.km /exp/rwicks/ersatz/data/raw/lt/dev/news-commentary/segmented.tail.news.2019.lt.shuffled.deduped /exp/rwicks/ersatz/data/raw/lv/dev/news-commentary/segmented.tail.news.2019.lv.shuffled.deduped /exp/rwicks/ersatz/data/raw/pl/dev/news-commentary/segmented.tail.news.2019.pl.shuffled.deduped /exp/rwicks/ersatz/data/raw/ps/dev/wikipedia/segmented.wikipedia.dev.ps-en.ps /exp/rwicks/ersatz/data/raw/ro/dev/news-commentary/segmented.tail.news.2019.ro.shuffled.deduped /exp/rwicks/ersatz/data/raw/ru/dev/wmt/wmt18.ru-en.ru /exp/rwicks/ersatz/data/raw/ta/dev/news-commentary/segmented.tail.news.2019.ta.shuffled.deduped /exp/rwicks/ersatz/data/raw/tr/dev/wmt/wmt16.tr-en.tr")
}

task make_train_data : ersatz
	:: vocab_path=@ left_size=@ right_size=@ train_data_path=@
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
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
    :: pyenv=@ :: .submitter=$grid :: devices=@
    :: devices_per_task=0
    :: .resource_flags=$cpuResourceFlags :: .action_flags=$cpuActionFlags
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


