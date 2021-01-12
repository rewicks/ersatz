global {
	batch_size=25000
	min_epochs=25
	max_epochs=1000
	lr=0.0001
	dropout=0.1
	log_interval=1000
	validation_interval=25000
	left_size=(LeftSize: 1)
	right_size=(RightSize: 1)
	embed_size=(EmbedSize: 16)
	transformer_nlayers=(TransformerLayers: 0 2)
    linear_nlayers=(LinearLayers: 0)
	activation_type=(ActivationType: tanh)
	nhead=(NHead: 8)
	eos_weight=(EOSWeight: 10.0)
	vocabulary_path=(Vocabulary: en_16000_vocab="/exp/rwicks/ersatz/data/vocab/wikimatrix.16000.en.model")

}
