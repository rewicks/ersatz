global {
	log_dir="/exp/rwicks/ersatz/exp/runs"
    batch_size=250000
	min_epochs=10
	max_epochs=1000
	lr=0.0001
	dropout=0.1
	log_interval=1000
	validation_interval=250000
    early_stopping=15
	left_size=(LeftSize: 1 2 4 6 8)
	right_size=(RightSize: 1 2 4 6)
	embed_size=(EmbedSize: 16 32 64 128 256 512)
	factor_embed_size=(FactSize: 0 8 16)
	transformer_nlayers=(TransformerLayers: 0 2 4)
    linear_nlayers=(LinearLayers: 0 1 2)
	activation_type=(ActivationType: tanh)
	nhead=(NHead: 8)
	eos_weight=(EOSWeight: 10.0 20.0 1.0 5.0 0.1 0.2)
	vocab_path=(Vocabulary: v500="/exp/rwicks/models/spm/unigram/ersatz.500."
							v12000="/exp/rwicks/models/spm/unigram/ersatz.12000.")
}
