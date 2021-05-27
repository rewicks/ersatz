global {
	log_dir="/exp/rwicks/ersatz/exp/runs"
    batch_size=25000
	min_epochs=5
	max_epochs=1000
	lr=(LearningRate: 0.0001 0.00001)
	dropout=0.1
	log_interval=100
	validation_interval=500
    early_stopping=15
	left_size=(LeftSize: 1 2 3 4 5 6)
	right_size=(RightSize: 1 2 3 4 5 6)
	embed_size=(EmbedSize: 16 32 64 128 256)
    factor_embed_size=(FactSize: 0 8 16 32)
	transformer_nlayers=(TransformerLayers: 0 1 2)
    linear_nlayers=(LinearLayers: 0 1 2 3)
	activation_type=(ActivationType: tanh)
	nhead=(NHead: 8)
	eos_weight=(EOSWeight: 10.0 20.0 1.0 5.0 0.1 0.2)
	vocab_size=(Vocabulary: 125 500 1000 4000 8000 12000)
}
