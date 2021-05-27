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
	reach score via (Scenario: en) *
		(Vocabulary: 500) *
        (LeftSize: 6) *
		(RightSize: 4) *
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
	reach score_dev via (Scenario: cs de en es et fi fr kk lt lv pl ro ru tr) *
        (LearningRate: 0.0001) *
		(Vocabulary: 500 1000) *
        (LeftSize: 6 8) *
		(RightSize: 4 6) *
		(EmbedSize: 64 128 256) *
        (FactSize: 0) *
		(TransformerLayers: 2) *
        (LinearLayers: 1) *
		(ActivationType: tanh) *
        (EOSWeight: 1.0 10.0 20.0)
    reach score_dev via (Scenario: ar gu hi iu km ps ta) *
        (LearningRate: 0.0001) *
		(Vocabulary: 500 1000) *
        (LeftSize: 6 8) *
		(RightSize: 4 6) *
		(EmbedSize: 128) *
        (FactSize: 0) *
		(TransformerLayers: 2) *
        (LinearLayers: 1) *
		(ActivationType: tanh) *
        (EOSWeight: 1.0 20.0)
    reach score_dev via (Scenario: ja) *
        (LearningRate: 0.0001) *
        (Vocabulary: 4000 8000) *
        (LeftSize: 6 8) *
        (RightSize: 4 6) *
        (EmbedSize: 128) *
        (FactSize: 0) *
        (TransformerLayers: 2) *
        (LinearLayers: 1) *
        (ActivationType: tanh) *
        (EOSWeight: 1.0 20.0)
    reach score_dev via (Scenario: zh) *
        (LearningRate: 0.0001) *
        (Vocabulary: 8000) *
        (LeftSize: 6 8) *
        (RightSize: 4 6) *
        (EmbedSize: 128) *
        (FactSize: 0) *
        (TransformerLayers: 2) *
        (LinearLayers: 1) *
        (ActivationType: tanh) *
        (EOSWeight: 1.0 20.0)
}

plan do_ml {
    reach train via (Scenario: multilingual) *
                (Vocabulary: 12000) *
                (LeftSize: 6) *
                (RightSize: 4) *
                (EmbedSize: 128 256) *
                (FactSize: 0) *
                (TransformerLayers: 1 2) *
                (LinearLayers: 1) *
                (ActivationType: tanh) *
                (EOSWeight: 10.0 20.0) *
                (LearningRate: 0.0001)
    reach train via (Scenario: multilingual) *
                (Vocabulary: 12000) *
                (LeftSize: 6) *
                (RightSize: 4) *
                (EmbedSize: 128 256) *
                (FactSize: 0) *
                (TransformerLayers: 0) *
                (LinearLayers: 2 3) *
                (ActivationType: tanh) *
                (EOSWeight: 10.0 20.0) *
                (LearningRate: 0.0001)
}

plan do_ml_context_grid {
    reach train via (Scenario: multilingual) *
                (Vocabulary: 12000) *
                (LeftSize: 1 2 3 4 5 6) *
                (RightSize: 1 2 3 4 5 6) *
                (EmbedSize: 128) *
                (FactSize: 0) *
                (TransformerLayers: 2) *
                (LinearLayers: 1) *
                (ActivationType: tanh) *
                (EOSWeight: 20.0) *
                (LearningRate: 0.0001) 
}


plan do_time {
    reach speed_test_cpu via (Scenario: multilingual) *
                (Vocabulary: 12000) *
                (LeftSize: 6) *
                (RightSize: 4) *
                (EmbedSize: 128) *
                (FactSize: 0) *
                (TransformerLayers: 1 2) *
                (LinearLayers: 1) *
                (ActivationType: tanh) *
                (EOSWeight: 20.0) *
                (LearningRate: 0.0001)
    reach speed_test_cpu via (Scenario: multilingual) *
                (Vocabulary: 12000) *
                (LeftSize: 6) *
                (RightSize: 4) *
                (EmbedSize: 128) *
                (FactSize: 0) *
                (TransformerLayers: 0) *
                (LinearLayers: 2 3) *
                (ActivationType: tanh) *
                (EOSWeight: 20.0) *
                (LearningRate: 0.0001)
}

plan do_score {
    reach score_dev via (Scenario: multilingual) *
                (Vocabulary: 12000) *
                (LeftSize: 6) *
                (RightSize: 4) *
                (EmbedSize: 128) *
                (FactSize: 0) *
                (TransformerLayers: 2) *
                (LinearLayers: 1) *
                (ActivationType: tanh) *
                (EOSWeight: 10.0) *
                (LearningRate: 0.0001)
}

plan do_baselines {
    reach score_baseline via (Scenario: multilingual) *
                (Base: *)
}
