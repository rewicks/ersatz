global {
	batch_size=25000
	min_epochs=25
	max_epochs=1000
	lr=0.0001
	dropout=0.1
	log_interval=1000
	validation_interval=25000
	left_size=(LeftSize: 1 2 4 6)
	right_size=(RightSize: 1 2 4 6)
	embed_size=(EmbedSize: 16 32 64)
	transformer_nlayers=(TransformerLayers: 0 2)
    linear_nlayers=(LinearLayers: 0 1 2)
	activation_type=(ActivationType: tanh)
	nhead=(NHead: 8)
	eos_weight=(EOSWeight: 10.0)
	vocab_path=(Vocabulary: en_16000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.16000.en.model"
                            en_8000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.8000.en.model"
                            en_2000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.2000.en.model"
                            en_500_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.500.en.model"
                            en_125_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.125.en.model")
}
