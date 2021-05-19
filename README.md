# Splitting with a Pre-Trained Model

1. Expects a `model_path` (should probably change to a default in expected folder location...)
2. `ersatz` reads from either stdin or a file path (via `--input`).
3. `ersatz` writes to either stdout or a file path (via `--output`).
4. An alternate candidate set for splitting may be given using `--determiner_type`
    * `multilingual` (default) is as described in paper
    * `en` requires a space following punctuation
    * `all` a space between any two characters
    * Custom can be written that uses the `determiner.Split()` base class
5. By default, expects raw sentences. Splitting a `.tsv` is also a supported behavior.
    1. `--text_ids` expects a comma separated list of column indices to split
    2. `--delim` changes the delimiter character (default is `\t`)
6. Uses gpu if available, to force cpu, use `--cpu`

### Example usage
Typical python usage:
```angular2html
python split.py --input unsegmented.txt --output sentences.txt ersatz.model
```

std[in,out] usage:
```angular2html
cat unsegmented.txt | split.py ersatz.model > sentences.txt
```
To split `.tsv` file:
```angular2html
cat unsegmented.tsv | split.py ersatz.model --text_ids 1 > sentences.txt
```

# Scoring a Model's Output

```angular2html
python score.py [gold_standard_file_path] [file_to_score]
```

(There are legacy arguments, but they're not used)

# Training a Model

## Data Preprocessing

### Vocabulary
Requires uses a pretrained `sentencepiece` model that has had `--eos_piece` replaced with `<eos>` and `--bos_piece` replaced with `<mos>`.

```angular2html
spm_train_py --input $TRAIN_DATA_PATH \
   --input_sentence_size 10000000 \
   --model_prefix ersatz \
   --vocab_size $VOCAB_SIZE \
   --bos_piece "<mos>" \
   --eos_piece "<eos>"
```

### Create training data

This pipeline takes a raw text file with one sentence per line (to use as labels) and creates a new raw text file
with the appropriate left/right context and labels. One line is one training example. User is expected to shuffle this
file manually (ie via `shuf`) after creation.

1. To create:
```angular2html
python dataset.py \
    --sentencepiece_path $SPM_PATH \
    --left-size $LEFT_SIZE \
    --right-size $RIGHT_SIZE \
    --output_path $OUTPUT_PATH \
    $INPUT_TRAIN_FILE_PATHS


shuf $OUTPUT_PATH > $SHUFFLED_TRAIN_OUTPUT_PATH
```
2. Repeat for validation data 
```angular2html
python dataset.py \
    --sentencepiece_path $SPM_PATH \
    --left-size $LEFT_SIZE \
    --right-size $RIGHT_SIZE \
    --output_path $VALIDATION_OUTPUT_PATH \
    $INPUT_DEV_FILE_PATHS
```

## Training
Something like: 

```angular2html
        python trainer.py \
        --sentencepiece_path=$vocab_path \
        --left_size=$left_size \
        --right_size=$right_size \
        --output_path=$out \
        --transformer_nlayers=$transformer_nlayers \
        --activation_type=$activation_type \
        --linear_nlayers=$linear_nlayers \
        --min-epochs=$min_epochs \
        --max-epochs=$max_epochs \
        --lr=$lr \
        --dropout=$dropout \
        --embed_size=$embed_size \
        --factor_embed_size=$factor_embed_size \
        --source_factors \
        --nhead=$nhead \
        --log_interval=$log_interval \
        --validation_interval=$validation_interval \
        --eos_weight=$eos_weight \
        --early_stopping=$early_stopping \
        --tb_dir=$LOGDIR \
        $train_path \
        $valid_path
```
