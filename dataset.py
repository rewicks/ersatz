import argparse
import os
from collections import namedtuple
import torch
import random
import sentencepiece as spm

#def other_split_train_file(file_path, eos=False, context_size=4):
#    content = open(file_path).read()
#    content = content.replace('\n', ' <eos> ')
#    content = content.split()
#    
#    window = ['<PAD>' for x in range(context_size)] + [content[:context_size+1]]
#    left_contexts = [window[:context_size]]
#    labels = [window[context_size]]
#    right_contexts = [window[context_size+1:]]
#
#    for index, word in enumerate(content):
#        if word != '<eos>' or eos:
#            window.pop(0)
#            window.append(word)
#            left_contexts.append(window[:context_size])
#            labels.append(window[context_size])
#            right_contexts.append(window[context_size+1:])
        
def page_generator(file_path):
    page = ''
    with open(file_path) as input_file:
        for line in input_file:
            if line != '\n':
                page += line
            else:
                page = page.replace('\n', ' <eos> ')
                page = page.split()
                yield page
                page = ''

def split_train_file(file_path, eos=False, left_context_size=5, right_context_size=5, eos_percent=0.25, sub_sample_percent=0.1 ):
    labels = []

    #content = open(file_path).read()
    #content = content.replace('\n', ' <eos> ')
    #content = content.split()

    random.seed(14)

    output_path = file_path.split('/')[-1].split('.')
    output_path = output_path[:-1] + [f'{left_context_size}-{right_context_size}-context'] + output_path[-1:]
    output_path = os.path.join('/'.join(file_path.split('/')[:-1]),'.'.join(output_path))

    
    with open(output_path, 'w') as f:
        for content in page_generator(file_path):
            for index, word in enumerate(content):
                random_number = random.random()
                if (word != '<eos>' and random_number <= (eos_percent*sub_sample_percent)) or (word == '<eos>' and random_number < sub_sample_percent):
                    if ((content[index-1] != '<eos>') or eos) and (index < len(content)-1 and '\u2581' in content[index+1]):
                        left_temp = []
                        right_temp = []
                        # Get the left context
                        temp_index = index - 1
                        while (len(left_temp) < left_context_size):
                            if temp_index >= 0:
                                if not eos:
                                    if content[temp_index] != '<eos>':
                                        left_temp.append(content[temp_index])
                                else:
                                    left_temp.append(content[temp_index])
                            else:
                                left_temp.append('<PAD>')
                            temp_index -= 1
        
                        left_temp.reverse()

                        # Get the label
                        label = word
                        #labels.append(word)

                        # Get the right context
                        temp_index = index + 1
                        while (len(right_temp) < right_context_size):
                            if temp_index < len(content):
                                if content[temp_index] != '<eos>':
                                    right_temp.append(content[temp_index])
                            else:
                                right_temp.append('<PAD>') 
                            temp_index += 1
                        f.write(' '.join(left_temp) + ' ||| ' + ' '.join(right_temp) + ' ||| ' + label + '\n')

def split_test_file(file_path, left_context_size, right_context_size, spm_path='/home/hltcoe/rwicks/ersatz/exp/02/models/spm.model'):
    left_contexts = []
    right_contexts = []

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)

    content = open(file_path).read()
    content = sp.EncodeAsPieces(content)

    for index, word in enumerate(content):
        if '\u2581' in word:
            left_temp = []
            right_temp = []
            # Get the left context
            temp_index = index - 1
            while (len(left_temp) < left_context_size):
                if temp_index >= 0:
                    left_temp.append(content[temp_index])
                else:
                    left_temp.append('<PAD>')
                temp_index -= 1
        
            left_temp.reverse()
            left_contexts.append(left_temp)

            # Get the right context
            temp_index = index
            while (len(right_temp) < right_context_size):
                if temp_index < len(content):
                    right_temp.append(content[temp_index])
                else:
                    right_temp.append('<PAD>') 
                temp_index += 1

            right_contexts.append(right_temp)

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
    parser.add_argument('path')
    parser.add_argument('--left-size', type=int)
    parser.add_argument('--right-size', type=int)
    parser.add_argument('--word-frequency', type=float)
    
    args = parser.parse_args()
    
    split_train_file(args.path, left_context_size=args.left_size, right_context_size=args.right_size, eos_percent=args.word_frequency)
    #left, right, labels = split_train_file(args.path, left_context_size=args.left_size, right_context_size=args.right_size, eos_percent=args.word_frequency)
    #write_training_files(args.path, left, right, labels, left_context_size=args.left_size, right_context_size=args.right_size)

Batch = namedtuple("Batch", "contexts labels left_size right_size")

class Vocabulary():

    def __init__(self, file_path, hole=True):
        #self.itos = ['<HOLE>', '<PAD>']
        #self.stoi = {'<UNK>': 0, '<HOLE>': 1, '<PAD>': 2}
        self.itos = ['<unk>', '<hole>', '<pad>']
        self.stoi = {'<unk>': 0, '<hole>': 1, '<pad>': 2}
        self.hole = hole
        self.build_vocab(file_path)

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
        return self.stoi.get(word, 0)

    def get_word(self, embedding):
        return self.itos[embedding]

    def string_to_tensor(self, sequences):
        arr = []
        for seq in sequences:
            tens = []
            for s in seq:
                tens.append(self.embed_word(s))
            arr.append(tens)
        return torch.tensor(arr)

    def tensor_to_string(self, tensors):
        output = []
        for tens in tensors:
            o = []
            for t in tens:
                o.append(self.get_word(t))
            output.append(' '.join(o))
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
                tens.append([self.embed_word('<HOLE>')])
            for r in right:
                tens.append([self.embed_word(r)])
            con_arr.append(tens)
            lab_arr.append(self.embed_word(label))
        return torch.tensor(con_arr), torch.tensor(lab_arr)

class ErsatzTrainDataset():
    def __init__(self, train_path, device, batch_size, vocabulary_path, transform=None, hole=False, sub_size=25000000):
        self.vocab = Vocabulary(vocabulary_path)
        self.device = device
        self.size = 0
        self.sub_size = sub_size
    
        if not os.path.exists(train_path):
            raise Exception("path does not exist")
        
        self.preprocess(train_path)
        #self.file_count = 55
        #self.left_context_size = 5
        #self.right_context_size = 5
        self.train_path = 'preprocess/' + train_path.split('/')[-1]
        self.batchify(batch_size)
        

    def __len__(self):
        return self.size

    def preprocess(self, train_path):
        output_lines = []
        if not os.path.isdir('preprocess'):
            os.mkdir('preprocess')
        output_path = 'preprocess/' + train_path.split('/')[-1]
        self.file_count = 0
        with open(train_path) as f:
            for line in f:
                self.size += 1
                output_lines.append(line.strip())
                left, right, label = line.strip().split('|||')
                #for word in left.strip().split():
                #    self.vocab.add_word(word)
                #for word in right.strip().split():
                #    self.vocab.add_word(word)
                #self.vocab.add_word(label.strip())
                if len(output_lines) > self.sub_size:
                    random.shuffle(output_lines)
                    with open(f'{output_path}.{self.file_count}.processed', 'w') as f:
                        f.write('\n'.join(output_lines))
                        self.file_count += 1
                    self.left_context_size = len(output_lines[0].split('|||')[0].strip().split())
                    self.right_context_size = len(output_lines[0].split('|||')[1].strip().split())
                    output_lines = []
        #self.context_size = len(output_lines[0].split('|||')[0].strip().split())*2 + 1

        if len(output_lines) > 0:
            random.shuffle(output_lines)
            with open(f'{output_path}.{self.file_count}.processed', 'w') as f:
                f.write('\n'.join(output_lines))
                self.file_count += 1

    def batchify(self, bsz):
        #self.batches = []
        #data = []
        for i in range(self.file_count):
            data = []
            batches = []
            with open(f'{self.train_path}.{i}.processed') as f:
                for index, line in enumerate(f):
                    if len(data) == bsz:
                        context, label = self.vocab.context_to_tensor(data)
                        context = context.view(bsz, -1)
                        label = label.view(bsz)
                        batches.append(Batch(context, label, self.left_context_size, self.right_context_size))
                        data = []
                    else:
                        left, right, label = line.strip().split('|||')
                        data.append((left.strip().split(), right.strip().split(), label.strip()))
            torch.save(batches, f'{self.train_path}.{i}.bin')

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

