import "submitters.tape"
import "versioners.tape"

global {
	pyenv="conda:ersatz"
	ducttape_output="/exp/rwicks/ersatz/exp/ducttape"
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

plan do_english_context_grid {
	reach train via (Scenario: en) *
		(Vocabulary: en_8000_vocab en_2000_vocab en_500_vocab en_125_vocab) *
        (LeftSize: 2 4 6) *
		(RightSize: 2 4 6) *
		(EmbedSize: 16 32 64) *
		(TransformerLayers: 0 2) *
        (LinearLayers: 1 2) *
		(ActivationType: tanh)
}

plan do_restart {
    reach train via (Scenario: en) *
        (Vocabulary: en_2000_vocab en_500_vocab en_125_vocab) *
        (LeftSize: 2 4 6) *
        (RightSize: 2 4) *
        (EmbedSize: 32 64 128) *
        (TransformerLayers: 2) *
        (LinearLayers: 1) *
        (ActivationType: tanh)
    reach train via (Scenario: en) *
        (Vocabulary: en_2000_vocab en_500_vocab en_125_vocab) *
        (LeftSize: 2 4 6) *
        (RightSize: 2 4) *
        (EmbedSize: 32 64 128) *
        (TransformerLayers: 0) *
        (LinearLayers: 2) *
        (ActivationType: tanh)
}

plan do_fine_grain {
    reach train via (Scenario: en) *
        (Vocabulary: en_500_vocab) *
        (LeftSize: 2 3 4 5 6 7 8) *
        (RightSize: 2 3 4 5) *
        (EmbedSize: 128) *
        (TransformerLayers: 2 4) *
        (LinearLayers: 1) *
        (ActivationType: tanh)
}


plan do_fine_grain_score {
    reach score via (Scenario: en) *
        (Vocabulary: en_500_vocab) *
        (LeftSize: 2 3 4 5 6 7 8) *
        (RightSize: 2 3 4 5) *
        (EmbedSize: 128) *
        (TransformerLayers: 2 4) *
        (LinearLayers: 1) *
        (ActivationType: tanh)
}


plan do_wikipedia {
    reach train via (Scenario: en_wp) *
        (Vocabulary: en_500_vocab) *
        (LeftSize: 6) *
        (RightSize: 4) *
        (EmbedSize: 128) *
        (TransformerLayers: 2) *
        (LinearLayers: 1) *
        (ActivationType: tanh)
}

plan do_shuffled {
    reach train via (Scenario: en_shuffled) *
        (Vocabulary: en_500_vocab) *
        (LeftSize: 6) *
        (RightSize: 4) *
        (EmbedSize: 128) *
        (TransformerLayers: 2) *
        (LinearLayers: 1) *
        (ActivationType: tanh) *
        (EOSWeight: 1.0 5.0 10.0 15.0 20.0 25.0 30.0)
}

plan do_ml_liz {
    reach train via (Scenario: multilingual) *
                (Vocabulary: ml_3000_vocab ml_48000_vocab) *
                (LeftSize: 4 6) *
                (RightSize: 4) *
                (EmbedSize: 32 128) *
                (TransformerLayers: 2) *
                (LinearLayers: 1) *
                (ActivationType: tanh) *
                (EOSWeight: 20.0)
    reach train via (Scenario: multilingual) *
                (Vocabulary: ml_3000_vocab ml_48000_vocab) *
                (LeftSize: 4 6) *
                (RightSize: 4) *
                (EmbedSize: 32 128) *
                (TransformerLayers: 0) *
                (LinearLayers: 2) *
                (ActivationType: tanh) *
                (EOSWeight: 20.0)
}


plan do_ml_rachel_1 {
    reach train via (Scenario: multilingual) *
                (Vocabulary: ml_12000_vocab) *
                (LeftSize: 4 6) *
                (RightSize: 4) *
                (EmbedSize: 32 128) *
                (TransformerLayers: 2) *
                (LinearLayers: 1) *
                (ActivationType: tanh) *
                (EOSWeight: 20.0)
    reach train via (Scenario: multilingual) *
                (Vocabulary: ml_12000_vocab) *
                (LeftSize: 4 6) *
                (RightSize: 4) *
                (EmbedSize: 32 128) *
                (TransformerLayers: 0) *
                (LinearLayers: 2) *
                (ActivationType: tanh) *
                (EOSWeight: 20.0)
}


plan do_weights {
    reach train via (Scenario: en) *
        (Vocabulary: en_2000_vocab) *
        (LeftSize: 4) *
        (RightSize: 2) *
        (EmbedSize: 128) *
        (TransformerLayers: 2) *
        (LinearLayers: 1) *
        (ActivationType: tanh) *
        (EOSWeight: 1.0)
}


plan do_english_char_context_grid {
	reach train via (Scenario: char_en) *
		(LeftSize: 1 2 3 4 5 6 7 8 9 10) *
		(RightSize: 1 2 3 4 5 6 7 8 9 10) *
		(EmbedSize: 256) *
		(TransformerLayers: 2) *
        (LinearLayers: 0) *
		(ActivationType: tanh) *
		(EmbedSize: 256)
}


#plan do_english_scoring {
#	reach score via (Scenario: en) *
#	    (Vocabulary: en_2000_vocab) *
#        (LeftSize: 2) *
#        (RightSize: 2) *
#        (EmbedSize: 32) *
#        (TransformerLayers: 2) *
#        (LinearLayers: 1) *
#        (ActivationType: tanh)
#}

plan do_english_scoring {
	reach score via (Scenario: en) *
	    (Vocabulary: en_2000_vocab en_500_vocab en_125_vocab) *
        (LeftSize: 2 4 6) *
        (RightSize: 2 4) *
        (EmbedSize: 32 64 128) *
        (TransformerLayers: 2) *
        (LinearLayers: 1) *
        (ActivationType: tanh)
    reach score via (Scenario: en) *
        (Vocabulary: en_2000_vocab en_500_vocab en_125_vocab) *
        (LeftSize: 2 4 6) *
        (RightSize: 2 4) *
        (EmbedSize: 32 64 128) *
        (TransformerLayers: 0) *
        (LinearLayers: 2) *
        (ActivationType: tanh)
}

plan do_ml_score {
    reach score via (Scenario: multilingual) *
                (Vocabulary: ml_12000_vocab) *
                (LeftSize: 4 6) *
                (RightSize: 4) *
                (EmbedSize: 32 128) *
                (TransformerLayers: 2) *
                (LinearLayers: 1) *
                (ActivationType: tanh) *
                (EOSWeight: 20.0)
    reach score via (Scenario: multilingual) *
                (Vocabulary: ml_12000_vocab) *
                (LeftSize: 4 6) *
                (RightSize: 4) *
                (EmbedSize: 32 128) *
                (TransformerLayers: 0) *
                (LinearLayers: 2) *
                (ActivationType: tanh) *
                (EOSWeight: 20.0)
}
