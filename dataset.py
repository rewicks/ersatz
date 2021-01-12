import argparse
import os
from collections import namedtuple
import torch
import random
import json
from subword import Vocabulary, SentencePiece

# iterates over a file and yields one doc at a time with the appropriately
# labelled splits; documents are separated by empty lines
def page_generator(file_path, tokenizer=None):
    page = []
    with open(file_path) as input_file:
        for line in input_file:
            if line != '\n':
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

                            f.write(' '.join(left_temp) + ' ||| ' + ' '.join(right_temp) + ' ||| ' + label + '\n')


# split test files
# the difference between this and the previous is there are no labels in data
def split_test_file(file_path, tokenizer, left_context_size, right_context_size):
    docs = []
    content = open(file_path).read().split('\n')
    content = tokenizer.encode(content, out_type=str)
    for c in content:
        left_contexts = []
        right_contexts = []
        for index, word in enumerate(c, 0):
            left_temp = []
            right_temp = []
            # Get the left context
            temp_index = index - 1
            while (len(left_temp) < left_context_size):
                if temp_index >= 0:
                    left_temp.append(c[temp_index])
                else:
                    left_temp.append('<pad>')
                temp_index -= 1
 
            left_temp.reverse()
            left_contexts.append(' '.join(left_temp))

            # Get the right context
            temp_index = index
            while (len(right_temp) < right_context_size):
                if temp_index < len(c):
                    right_temp.append(c[temp_index])
                else:
                    right_temp.append('<pad>') 
                temp_index += 1

            right_contexts.append(' '.join(right_temp))
        docs.append((left_contexts, right_contexts))
    return docs



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
                    data.append((left.strip(), right.strip(), label.strip()))
                    context_strings.append((left.strip(), right.strip()))
                if len(data) >= batch_size:
                    context, label = self.tokenizer.context_to_tensor(data)
                    context = context.view(len(data), -1)
                    label = label.view(len(data))
                    yield context, label, context_strings
                    batch_idx += 1
                    data = []
                    context_strings = []
            context, label = self.tokenizer.context_to_tensor(data)
            context = context.view(len(data), -1)
            label = label.view(len(data))
            if len(data) > 0:
                yield context, label, context_strings
                batch_idx += 1

