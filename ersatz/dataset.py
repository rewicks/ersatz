import os
import random
import string
import pathlib
import sys
import logging
import argparse

if __package__ is None and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'ersatz'

from .subword import Vocabulary, SentencePiece
from .candidates import MultilingualPunctuation, PunctuationSpace, Split

logger = logging.getLogger('ersatz')

#######################################################################################################

# iterates over a file and yields one doc at a time with the appropriately
# labelled splits; documents are separated by empty lines
def document_generator(file_path, tokenizer=None):
    document = []
    with open(file_path) as input_file:
        for line in input_file:
            if len(tokenizer.encode(line, out_type=str)) > 0:
                line = line.strip()
                line = tokenizer.encode(line, out_type=str)
                new_line = []
                for l in line:
                    new_line.append(l)
                    new_line.append('<mos>')
                new_line[-1] = '<eos>'
                document += new_line
            else:
                yield document
                document = []
    yield document

# this builds all the training data from a plain text file
# writes it out to a new file
def split_train_file(file_paths,
                     tokenizer,
                     output_path=None,
                     left_context_size=5,
                     right_context_size=5,
                     determiner=None):

    random.seed(14)

    # import pdb; pdb.set_trace()
    with open(output_path, 'w') as f:
        for file_path in file_paths:
            for doc in document_generator(file_path, tokenizer=tokenizer):
                if len(doc) > 0:
                    left_temp = ["<pad>" for x in range(left_context_size-1)] + [doc[0]]
                    right_temp = [x for x in doc[1:(2*right_context_size)+1] if x not in ["<eos>", "<mos>"]]
                    temp_index = 2*right_context_size+2
                    for index, word in enumerate(doc):
                        if word in ['<eos>', '<mos>']:

                            label = word

                            if determiner(''.join(left_temp).replace('\u2581', ' ').replace('<pad>', ''),
                                          ''.join(right_temp).replace('\u2581', ' ').replace('<pad>', '')):
                                f.write(' '.join(left_temp) + ' ||| ' + ' '.join(right_temp) + ' ||| ' + label + '\n')

                            left_temp.pop(0)
                            left_temp.append(right_temp.pop(0))
                            if temp_index < len(doc):
                                right_temp.append(doc[temp_index])
                                temp_index += 2
                            else:
                                right_temp.append("<pad>")
# split test files
# the difference between this and the previous is there are no labels in data
def split_test_file(document, tokenizer, left_context_size, right_context_size):
    document = tokenizer.encode(document, out_type=str)
    left_contexts = []
    right_contexts = []
    if len(document) > 0:
        left_temp = ["<pad>" for x in range(left_context_size - 1)] + [document[0]]
        right_temp = [x for x in document[1:(right_context_size) + 1]]
        while (len(right_temp) < right_context_size):
            right_temp.append("<pad>")
        temp_index = right_context_size + 1
        for index, word in enumerate(document, 0):
            left_contexts.append(' '.join(left_temp))

            right_contexts.append(' '.join(right_temp))

            left_temp.pop(0)
            left_temp.append(right_temp.pop(0))
            if temp_index < len(document):
                right_temp.append(document[temp_index])
                temp_index += 1
            else:
                right_temp.append("<pad>")
    return left_contexts, right_contexts



def write_training_files(file_path, left_contexts, right_contexts, labels, left_context_size=5, right_context_size=5):
    output_path = file_path.split('/')[-1].split('.')
    output_path = output_path[:-1] + [f'{left_context_size}-{right_context_size}-context'] + output_path[-1:]
    output_path = os.path.join('/'.join(file_path.split('/')[:-1]),'.'.join(output_path))

    with open(output_path, 'w') as f:
        for left, right, label in zip(left_contexts, right_contexts, labels):
            f.write(' '.join(left) + ' ||| ' + ' '.join(right) + ' ||| ' + label + '\n')

class SourceFactors():
    def __init__(self):
        self.codes = {
            'UNMARK': 0,
            'CAP': 1,
            'LOWER': 2,
            'PUNC': 3,
            'TITLE': 4,
            'NUMBER': 5
        }
        pass

    # specific to sentencepiece
    def compute(self, token_stream):
        word = []
        output_stream = []
        token_stream = token_stream.split()
        for t in token_stream + ['\u2581']:
            if '\u2581' in t:
                out = None
                # potentially add a marker for truncated words in left context
                if len(word) > 0:
                    untok = ''.join(word).replace('\u2581', '')
                    if untok.istitle():
                        out = [self.codes['TITLE'] for w in word]
                    elif untok.isupper():
                        out = [self.codes['CAP'] for w in word]
                    elif untok.islower():
                        out = [self.codes['LOWER'] for w in word]
                    elif untok in string.punctuation:
                        out = [self.codes['PUNC'] for w in word]
                    else:
                        for w in untok:
                            if w in string.digits:
                                out = [self.codes['NUMBER'] for w in word]
                                break
                        if not out:
                            out = [self.codes['UNMARK'] for w in word]
                    output_stream += out
                word = []
            word.append(t)
        assert(len(output_stream)==len(token_stream))
        return output_stream


class ErsatzDataset():
    def __init__(self, data_path, device,
                 left_context_size=15,
                 right_context_size=5,
                 sentencepiece_path=None,
                 tokenizer=None):
        if tokenizer is None:
            if sentencepiece_path is not None:
                self.tokenizer = SentencePiece(model_path=sentencepiece_path)
            else:
                self.tokenizer = Vocabulary()
                self.tokenizer.build_vocab(data_path)
        else:
            self.tokenizer = tokenizer
        self.device = device
        self.size = 0
    
        if not os.path.exists(data_path):
            raise Exception("path does not exist")

        self.left_context_size = left_context_size
        self.right_context_size = right_context_size

        self.data_path = data_path
        self.source_factors = SourceFactors()

    def __len__(self):
        return self.size

    def batchify(self, batch_size):
        data = []
        context_strings = []
        batch_idx = 0
        #factors = []
        with open(self.data_path) as f:
            for line in f:
                self.size += 1
                if len(line.strip().split('|||')) == 3:
                    left, right, label = line.strip().split('|||')
                    # little check because some datasets have '|||' ... maybe change eventually to special character code ?
                    if (len(left.split()) == self.left_context_size) and (len(right.split()) == self.right_context_size):
                        data.append((left.strip(), self.source_factors.compute(left.strip()),
                                     right.strip(), self.source_factors.compute(right.strip()),
                                     label.strip()))
                        context_strings.append((left.strip(), right.strip()))
                if len(data) >= batch_size:
                    context, factors, label = self.tokenizer.context_to_tensor(data)
                    context = context.view(len(data), -1)
                    factors = factors.view(len(data), -1)
                    label = label.view(len(data))
                    yield context, factors, label, context_strings
                    batch_idx += 1
                    data = []
                    context_strings = []
            context, factors, label = self.tokenizer.context_to_tensor(data)
            context = context.view(len(data), -1)
            factors = factors.view(len(data), -1)
            label = label.view(len(data))
            if len(data) > 0:
                yield context, factors, label, context_strings
                batch_idx += 1

##############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="ERSATZ PREPROCESSOR: converts raw text (~one sentence per line) to expected input for ersatz training.\n"
        "      Example: ersatz_preprocess --sp en.8000.model --output_path en.train file1.txt file2.txt file3.txt",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--sentencepiece_path', '--sp', type=str, default=None,
                        help="Path to sentencepiece .model file to be used as tokenizer")
    parser.add_argument('--output_path', type=str, default="train.data",
                        help="File path where output will be written")
    parser.add_argument('--left-size', type=int, default=5,
                        help="Number of tokens of left context to use for predictions")
    parser.add_argument('--right-size', type=int, default=5,
                        help="Number of tokens of right context to use for predictions")
    parser.add_argument('--determiner_type', default='multilingual', choices=["en", "multilingual", "all"],
                        help="Type of contexts to include. Defaults to 'multilingual'\n"
                            "   * en: [EOS punctuation][any_punctuation]*[space]\n"
                            "   * multilingual: [EOS punctuation][!number]\n"
                            "   * all: all possible contexts")
    parser.add_argument('--input_paths', nargs='*', default=None,
                        help="Paths to raw text input files")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.sentencepiece_path is not None:
        tokenizer = SentencePiece(model_path=args.sentencepiece_path)
    else:
        logger.error("ERROR: No --sentencepiece_path was given. Training one as part of preprocessing is not currently supported.")
        sys.exit(-1)

    if args.determiner_type == "en":
        determiner = PunctuationSpace()
    elif args.determiner_type == "multilingual":
        determiner = MultilingualPunctuation()
    else:
        determiner = Split()

    split_train_file(args.input_paths, tokenizer,
                     output_path=args.output_path,
                     left_context_size=args.left_size,
                     right_context_size=args.right_size,
                     determiner=determiner)


if __name__ == '__main__':
    main()
