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
