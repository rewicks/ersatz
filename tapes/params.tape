global {
	batch_size=25000
	min_epochs=25
	max_epochs=1000
	lr=0.0001
	dropout=0.1
	log_interval=1000
	validation_interval=25000
	left_size=(LeftSize: 1 2 6 8)
	right_size=(RightSize: 1 2 4 6)
	embed_size=(EmbedSize: 16 32 64 128 256 512)
	transformer_nlayers=(TransformerLayers: 0 2 4)
    linear_nlayers=(LinearLayers: 0 1 2)
	activation_type=(ActivationType: tanh)
	nhead=(NHead: 8)
	eos_weight=(EOSWeight: 10.0 20.0 1.0 5.0 15.0 20.0 25.0 30.0 40.0 50.0 60.0)
	vocab_path=(Vocabulary: en_16000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.16000.en.model"
                            en_125_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.125.en.model"
                            en_500_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.500.en.model"
                            ml_3000_vocab="/exp/rwicks/models/spm/unigram/wikimatrix.3000.wmtlangs.model"
                            en_2000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.2000.en.model"
                            ml_12000_vocab="/exp/rwicks/models/spm/unigram/wikimatrix.12000.wmtlangs.model"
                            ml_48000_vocab="/exp/rwicks/models/spm/unigram/wikimatrix.48000.wmtlangs.model")
}
