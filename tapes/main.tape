import "submitters.tape"
import "versioners.tape"

global {
	pyenv="conda:ersatz"
	ducttape_output="/exp/rwicks/ersatz/exp/ducttape"
	ducttape_experimental_packages=true
	ducttape_experimental_submitters=true
	ducttape_experimental_imports=true
	ducttape_experimental_multiproc=true
}

import "env_coe.tape"
import "params.tape"
import "prepro.tape"
import "train.tape"

plan do_english_context_grid {
	reach train via (Scenario: en) *
		(LeftSize: 1) *
		(RightSize: 1) *
		(EmbedSize: 256) *
		(ModelType: transformer) *
		(ActivationType: tanh) *
		(EmbedSize: 256)
}

plan do_english_char_context_grid {
	reach train via (Scenario: char_en) *
		(LeftSize: 1 2 3 4 5 6 7 8 9 10) *
		(RightSize: 1 2 3 4 5 6 7 8 9 10) *
		(EmbedSize: 256) *
		(ModelType: transformer) *
		(ActivationType: tanh) *
		(EmbedSize: 256)
}

