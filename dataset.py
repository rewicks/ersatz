import argparse
import os
from collections import namedtuple
import torch
import random, string
import json
from subword import Vocabulary, SentencePiece
from determiner import MultilingualPunctuation
# iterates over a file and yields one doc at a time with the appropriately
# labelled splits; documents are separated by empty lines
def page_generator(file_path, tokenizer=None):
    page = []
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
                page += new_line
            else:
                yield page
                page = []
    yield page

# this builds all the training data from a plain text file
# writes it out to a new file
def split_train_file(file_paths, tokenizer, output_path=None, left_context_size=5, right_context_size=5, eos_percent=0.25, sub_sample_percent=0.1):
    random.seed(14)

    det = MultilingualPunctuation()

    with open(output_path, 'w') as f:
        for file_path in file_paths:
            for content in page_generator(file_path, tokenizer=tokenizer):
                for index, word in enumerate(content):
                    random_number = random.random()
                    if (word == '<mos>' and random_number <= (eos_percent*sub_sample_percent)) or (word == '<eos>' and random_number <= sub_sample_percent):
                        if index < len(content)-1 and '\u2581' in content[index+1]:
                            left_temp = []
                            right_temp = []
                            
                            # Get the left context
                            temp_index = index-1
                            while (len(left_temp) < left_context_size):
                                if temp_index >= 0:
                                    if content[temp_index] not in ['<eos>', '<mos>']:
                                        left_temp.append(content[temp_index])
                                else:
                                    left_temp.append('<pad>')
                                temp_index -= 1
                            left_temp.reverse()
                    
                            label = word

                            temp_index = index + 1
                            while (len(right_temp) < right_context_size):
                                if temp_index < len(content): 
                                    if content[temp_index] not in ['<eos>', '<mos>']:
                                        right_temp.append(content[temp_index])
                                else:
                                    right_temp.append('<pad>')
                                temp_index += 1
        
                            if det(' '.join(left_temp), ' '.join(right_temp)):
                                f.write(' '.join(left_temp) + ' ||| ' + ' '.join(right_temp) + ' ||| ' + label + '\n')


# split test files
# the difference between this and the previous is there are no labels in data
def split_test_file(content, tokenizer, left_context_size, right_context_size):
    content = tokenizer.encode(content, out_type=str)
    left_contexts = []
    right_contexts = []
    for index, word in enumerate(content, 0):
        left_temp = []
        right_temp = []
        # Get the left context
        temp_index = index - 1
        while (len(left_temp) < left_context_size):
            if temp_index >= 0:
                left_temp.append(content[temp_index])
            else:
                left_temp.append('<pad>')
            temp_index -= 1

        left_temp.reverse()
        left_contexts.append(' '.join(left_temp))

        # Get the right context
        temp_index = index
        while (len(right_temp) < right_context_size):
            if temp_index < len(content):
                right_temp.append(content[temp_index])
            else:
                right_temp.append('<pad>')
            temp_index += 1

        right_contexts.append(' '.join(right_temp))
    return left_contexts, right_contexts



def write_training_files(file_path, left_contexts, right_contexts, labels, left_context_size=5, right_context_size=5):
    output_path = file_path.split('/')[-1].split('.')
    output_path = output_path[:-1] + [f'{left_context_size}-{right_context_size}-context'] + output_path[-1:]
    output_path = os.path.join('/'.join(file_path.split('/')[:-1]),'.'.join(output_path))

    with open(output_path, 'w') as f:
        for left, right, label in zip(left_contexts, right_contexts, labels):
            f.write(' '.join(left) + ' ||| ' + ' '.join(right) + ' ||| ' + label + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--sentencepiece_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--left-size', type=int)
    parser.add_argument('--right-size', type=int)
    parser.add_argument('--word-frequency', type=float, default=1.0)
    parser.add_argument('--subsample-frequency', type=float, default=1.0)
    args = parser.parse_args()

    if args.sentencepiece_path is not None:
        tokenizer = SentencePiece(args.sentencepiece_path)
    else:
        tokenizer = Vocabulary()
    split_train_file(args.paths, tokenizer,
                     output_path=args.output_path,
                     left_context_size=args.left_size,
                     right_context_size=args.right_size,
                     eos_percent=args.word_frequency,
                     sub_sample_percent=args.subsample_frequency)

class SourceFactors():
    def __init__(self):
        self.codes = {
            'TITLE': 0,
            'CAP': 1,
            'LOWER': 2,
            'PUNC': 3,
            'UNMARK': 4,
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
                 left_context_size=15, right_context_size=5,
                 sentencepiece_path=None,
                 tokenizer=None):
        if tokenizer is None:
            if sentencepiece_path is not None:
                self.tokenizer = SentencePiece(sentencepiece_path)
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
        with open(self.data_path) as f:
            for line in f:
                self.size += 1
                if len(line.strip().split('|||')) == 3:
                    left, right, label = line.strip().split('|||')
                    # little check because some datasets have '|||'....maybe change eventually to special character code?
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

