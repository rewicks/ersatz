import argparse
import os
from collections import namedtuple
import torch
import random
import json
#import sentencepiece as spm

PUNCTUATION = ['`', '!', '\'', '"', ';', ':', '.', '?', ','] 
        
def page_generator(file_path, sliding_window=10):
    page = ''
    with open(file_path) as input_file:
        for line in input_file:
            if line != '\n':
                page += line
            else:
                page = page.replace(' \u2581', ' <mos> \u2581')
                page = page.replace('\n', ' <eos> ')
                page = page.split()
                yield page
                page = ''

    page = page.replace(' \u2581', ' <mos> \u2581')
    page = page.replace('\n', ' <eos> ')
    page = page.split()
    yield page

def split_train_file(file_paths, output_path=None, eos=False, left_context_size=5, right_context_size=5, eos_percent=0.25, sub_sample_percent=0.1, punc_percent=0.0):
    labels = []

    random.seed(14)

    if output_path is None:
        output_path = file_path.split('/')[-1].split('.')
        output_path = output_path[:-1] + [f'{left_context_size}-{right_context_size}-context.p{punc_percent}'] + output_path[-1:]
        output_path = os.path.join('/'.join(file_path.split('/')[:-1]),'.'.join(output_path))

    with open(output_path, 'w') as f:
        for file_path in file_paths:
            for content in page_generator(file_path):
                for index, word in enumerate(content):
                    random_number = random.random()
                    punc_number = random.random()
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



def split_test_file(file_path, left_context_size, right_context_size, spm_path='/home/hltcoe/rwicks/ersatz/exp/02/models/spm.model'):
    docs = []

    #sp = spm.SentencePieceProcessor()
    #sp.Load(spm_path)

    content = open(file_path).read().split('\n')
    #content = sp.EncodeAsPieces(content)
    for c in content:
        c = c.strip().split()
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
            left_contexts.append(left_temp)

            # Get the right context
            temp_index = index
            while (len(right_temp) < right_context_size):
                if temp_index < len(c):
                    right_temp.append(c[temp_index])
                else:
                    right_temp.append('<pad>') 
                temp_index += 1

            right_contexts.append(right_temp)
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
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--left-size', type=int)
    parser.add_argument('--right-size', type=int)
    parser.add_argument('--word-frequency', type=float, default=1.0)
    parser.add_argument('--punc-frequency', type=float, default=1.0)
    parser.add_argument('--subsample-frequency', type=float, default=1.0)
    args = parser.parse_args()
    split_train_file(args.paths, output_path=args.output_path, left_context_size=args.left_size, right_context_size=args.right_size, eos_percent=args.word_frequency, punc_percent=args.punc_frequency, sub_sample_percent=args.subsample_frequency)
    #left, right, labels = split_train_file(args.path, left_context_size=args.left_size, right_context_size=args.right_size, eos_percent=args.word_frequency)
    #write_training_files(args.path, left, right, labels, left_context_size=args.left_size, right_context_size=args.right_size)

Batch = namedtuple("Batch", "contexts labels")

class Vocabulary():

    def __init__(self, vocab_path=None, hole=True):
        if vocab_path is None:
            self.itos = ['<eos>', '<mos>', '<unk>', '<hole>', '<pad>']
            self.stoi = {'<eos>': 0, '<mos>': 1, '<unk>': 2, '<hole>': 3, '<pad>': 4}
        else:
            self.stoi = json.load(open(vocab_path))
            self.itos = ['' for x in self.stoi]
            for key in self.stoi:
                self.itos[self.stoi[key]] = key
            for word in ['<eos>', '<mos>', '<unk>', '<hole>', '<pad>']:
                if word not in self.stoi:
                    self.add_word(word)      
       
        self.hole = hole

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, file_path):
        with open(file_path) as i:
            for line in i:
                word = line.split()[0].strip()
                self.add_word(word)

    def add_word(self, word):
        if word not in self.stoi:
            self.stoi[word] = len(self.itos)
            self.itos.append(word)
    
    def embed_word(self, word):
        return self.stoi.get(word, 2)

    def get_word(self, embedding):
        return self.itos[embedding]

    def detokenize(self, input_string):
        input_string = input_string.replace(' ', '')
        input_string = input_string.replace('\u2581', ' ')
        return input_string

    def string_to_tensor(self, sequences):
        arr = []
        for seq in sequences:
            tens = []
            for s in seq:
                tens.append(self.embed_word(s))
            arr.append(tens)
        return torch.tensor(arr)

    def tensor_to_string(self, tensors):
        if len(tensors.shape) > 1:
            output = []
            for tens in tensors:
                o = []
                for t in tens:
                    o.append(self.get_word(t))
                output.append(' '.join(o))
        else:
            o = []
            for t in tensors:
                o.append(self.get_word(t))
            output = ' '.join(o)
        return output

    def context_to_tensor(self, contexts):
        con_arr = []
        lab_arr = []
        for left, right, label in contexts:
            tens = []
            for l in left:
                tens.append([self.embed_word(l)])
            if self.hole:
                #print('appending hole')
                tens.append([self.embed_word('<hole>')])
            for r in right:
                tens.append([self.embed_word(r)])
            con_arr.append(tens)
            lab_arr.append(self.embed_word(label))
        return torch.tensor(con_arr), torch.tensor(lab_arr)

class ErsatzDataset():
    def __init__(self, data_path, device, left_context_size=15, right_context_size=5, vocabulary_path=None, vocab=None, transform=None, hole=False, sub_size=25000000):
        if vocabulary_path is not None:
            self.vocab = Vocabulary(vocab_path=vocabulary_path)
        elif vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary()
            self.vocab.build_vocab(data_path)
        self.device = device
        self.size = 0
        self.sub_size = sub_size
    
        if not os.path.exists(data_path):
            raise Exception("path does not exist")

        self.left_context_size = left_context_size
        self.right_context_size = right_context_size
        
        #self.preprocess(data_path)
        self.data_path = data_path
        #self.data_path = 'preprocess/' + data_path.split('/')[-1]
        #self.batchify(batch_size)
        

    def __len__(self):
        return self.size

    def batchify(self, batch_size):
        data = []
        batch_idx = 0
        with open(self.data_path) as f:
            for line in f:
                self.size += 1
                if len(line.strip().split('|||')) == 3:
                    left, right, label = line.strip().split('|||')
                    data.append((left.strip().split(), right.strip().split(), label.strip()))
                if len(data) >= batch_size:
                    context, label = self.vocab.context_to_tensor(data)
                    context = context.view(len(data), -1)
                    label = label.view(len(data)) 
                    yield batch_idx, Batch(context, label)
                    batch_idx += 1
                    data = []
            context, label = self.vocab.context_to_tensor(data)
            context = context.view(len(data), -1)
            label = label.view(len(data))
            if len(data) > 0:
                yield batch_idx, Batch(context, label)
                batch_idx += 1
    '''
    def batchify(self, bsz):
        #self.batches = []
        #data = []
        batch_idx = 0
        
        for i in range(self.file_count):
            data = []
            batches = []
            with open(f'{self.data_path}.{i}.processed') as f:
                for index, line in enumerate(f):
                    if len(data) >= bsz:
                        context, label = self.vocab.context_to_tensor(data)
                        context = context.view(bsz, -1)
                        label = label.view(bsz)
                        yield batch_idx, Batch(context, label, self.left_context_size, self.right_context_size)
                        batch_idx += 1
                        #batches.append(Batch(context, label, self.left_context_size, self.right_context_size))
                        data = []
                    else:
                        if len(line.strip().split('|||') == 3:
                            left, right, label = line.strip().split('|||')
                            data.append((left.strip().split(), right.strip().split(), label.strip()))

            context, label = self.vocab.context_to_tensor(data)
            context = context.view(len(data), -1)
            label = label.view(len(data))
            if len(data) > 0:
                yield batch_idx, Batch(context, label, self.left_context_size, self.right_context_size)
                batch_idx += 1
            #torch.save(batches, f'{self.data_path}.{i}.bin')
        '''
'''
class ErsatzValidDataset():
    def __init__(self, valid_path, device, batch_size, vocab, transform=None, hole=False):
        self.vocab = vocab
        self.device = device
        self.size = 0
    
        if not os.path.exists(valid_path):
            raise Exception("path does not exist")
        
        self.preprocess(valid_path)
        self.valid_path = 'preprocess/' + valid_path.split('/')[-1]
        self.batchify(valid_path, batch_size)
         

    def __len__(self):
        return self.size

    def preprocess(self, valid_path):
        output_lines = []
        if not os.path.isdir('preprocess'):
            os.mkdir('preprocess')
        output_path = 'preprocess/' + valid_path.split('/')[-1]
        with open(valid_path) as f:
            for line in f:
                self.size += 1
                output_lines.append(line.strip())
                left, right, label = line.strip().split('|||')
        self.left_context_size = len(output_lines[0].split('|||')[0].strip().split())
        self.right_context_size = len(output_lines[0].split('|||')[1].strip().split())

        random.shuffle(output_lines)
        with open(output_path, 'w') as f:
            f.write('\n'.join(output_lines))

    def batchify(self, valid_path, bsz):
        self.batches = []
        with open(self.valid_path) as f:
            data = []
            for index, line in enumerate(f):
                if len(data) == bsz:
                    context, label = self.vocab.context_to_tensor(data)
                    context = context.view(bsz, -1)
                    label = label.view(bsz)
                    self.batches.append(Batch(context, label, self.left_context_size, self.right_context_size))
                    data = []
                else:
                    left, right, label = line.strip().split('|||')
                    data.append((left.strip().split(), right.strip().split(), label.strip()))
'''
