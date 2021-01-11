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
	transformer_nlayers=(TransformerLayers: 0 1 2 3 4 5)
    linear_nlayers=(LinearLayers: 0 1 2 3 4 5 6)
	activation_type=(ActivationType: tanh)
	nhead=(NHead: 2 4 6 8)
	eos_weight=(EOSWeight: 10.0 15.0 20.0 25.0 30.0 35.0)
	vocabulary_path=(Vocabulary: en_16000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.10000.en.json"
                                en_8000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.8000.en.json"
                                en_4000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.4000.en.json"
				   				en_2000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.2000.en.json"
                                en_1000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.1000.en.json"
                                en_500_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.500.en.json"
                                en_250_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.250.en.json"
                                en_125_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.125.en.json"
                                multilingual_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.wmtlangs.50000.json")

}
