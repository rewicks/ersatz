global {
	batch_size=25000
	min_epochs=25
	max_epochs=1000
	lr=0.0001
	dropout=0.1
	log_interval=1000
	validation_interval=25000
	left_size=(LeftSize: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
	right_size=(RightSize: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
	embed_size=(EmbedSize: 16 32 64 128 256 512)
	model_type=(ModelType: transformer linear)
	activation_type=(ActivationType: tanh)
	nlayers=(NLayers: 1 2 3 4)
	nhead=(NHead: 2 4 6 8)
	eos_weight=(EOSWeight: 10.0 15.0 20.0 25.0 30.0 35.0)
	vocabulary_path=(Vocabulary: en_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.10000.en.json"
				   				multilingual_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.wmtlangs.50000.json")

}
