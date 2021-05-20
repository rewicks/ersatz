import "submitters.tape"
import "versioners.tape"

global {
	pyenv="conda:ersatz"
	ducttape_experimental_packages=true
	ducttape_experimental_submitters=true
	ducttape_experimental_imports=true
	ducttape_experimental_multiproc=true
    devices="0,1,2,3,4,5,6,7"
}

import "env_coe.tape"
import "params.tape"
import "prepro.tape"
import "train.tape"
import "test.tape"

plan do_english {
	ducttape_output="/exp/rwicks/ersatz/exp/english"
	reach score via (Scenario: en) *
		(Vocabulary: 500) *
        (LeftSize: 6) *
		(RightSize: 4 ) *
		(EmbedSize: 64) *
		(TransformerLayers: 2) *
        (LinearLayers: 1) *
		(ActivationType: tanh) *
        (EOSWeight: 1.0)
}

plan do_english_wikipedia {
	ducttape_output="/exp/rwicks/ersatz/exp/english_wikipedia"
	reach score via (Scenario: en) *
		(Vocabulary: 500) *
        (LeftSize: 6) *
		(RightSize: 4 ) *
		(EmbedSize: 64) *
		(TransformerLayers: 2) *
        (LinearLayers: 1) *
		(ActivationType: tanh) *
        (EOSWeight: 1.0)
}

plan do_shuffled {
    reach score via (Scenario: en_shuffled) *
        (Vocabulary: en_500_vocab) *
        (LeftSize: 6) *
        (RightSize: 4) *
        (EmbedSize: 128) *
        (TransformerLayers: 2) *
        (LinearLayers: 1) *
        (ActivationType: tanh) *
        (EOSWeight: 1.0 5.0 10.0 15.0 20.0 25.0)
}

plan do_monolinguals {
	reach score via (Scenario: ar cs de en es et fi fr gu hi iu kk km lt lv pl ps ro ru ta tr) *
		(Vocabulary: 500) *
        (LeftSize: 6) *
		(RightSize: 4) *
		(EmbedSize: 64 128) *
		(TransformerLayers: 2) *
        (LinearLayers: 1) *
		(ActivationType: tanh) *
        (EOSWeight: 1.0 5.0 10.0)
}

plan do_ml {
    reach score_dev via (Scenario: multilingual) *
                (Vocabulary: 12000) *
                (LeftSize: 6) *
                (RightSize: 4) *
                (EmbedSize: 128) *
                (FactSize: 0 16 32) *
                (TransformerLayers: 2) *
                (LinearLayers: 1) *
                (ActivationType: tanh) *
                (EOSWeight: 1.0 5.0 0.2 10.0 0.1) *
                (LearningRate: 0.0001 0.00001)
}

plan do_baselines {
    reach score_baseline via (Scenario: multilingual) *
                (Base: *)
}
