import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random
import os

Batch = namedtuple("Batch", "contexts labels size")

class Vocabulary():

    def __init__(self, hole=True):
        self.itos = ['<UNK>', '<HOLE>', '<PAD>']
        self.stoi = {'<UNK>': 0, '<HOLE>': 1, '<PAD>': 2}
        self.hole = hole

    def __len__(self):
        return len(self.itos)    

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
                tens.append([self.embed_word('<HOLE>')])
            for r in right:
                tens.append([self.embed_word(r)])
            con_arr.append(tens)
            lab_arr.append(self.embed_word(label))
        return torch.tensor(con_arr), torch.tensor(lab_arr)
 
class ErsatzTrainDataset():

    def __init__(self, train_path, device, transform=None, hole=False):
        self.train_lines = []
        if not os.path.exists(train_path):
            raise Exception("path does not exist")
        with open(train_path) as f:
            for line in f:
                left, right, label = line.strip().split('|||')
                self.train_lines.append((left.strip().split(), right.strip().split(), label.strip()))

        self.window_size = len(self.train_lines[0][0])
        self.context_size = (self.window_size*2) + 1
        self.vocab = Vocabulary()
        self.device = device

    def __len__(self):
        return len(self.train_lines)

    def build_vocab(self):
        for line in self.train_lines:
            left, right, label = line
            for word in left:
                self.vocab.add_word(word)
            for word in right:
                self.vocab.add_word(word)
            self.vocab.add_word(label) 
    
    def batchify(self, bsz):
        data, labels = self.vocab.context_to_tensor(self.train_lines)
        
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remaineders).
        data = data.narrow(0, 0, nbatch * bsz)
        labels = labels.narrow(0, 0, nbatch * bsz)       

        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        labels = labels.view(bsz, -1).t().contiguous()

        self.batches = []
        data = data.view(-1, self.context_size, bsz)
        labels = labels.view(-1, 1, bsz)
        for context_batch, label_batch, in zip(data, labels):
            self.batches.append(Batch(context_batch.t(), label_batch[0], self.context_size))

        random.shuffle(self.batches)


class TransformerModel(nn.Module):
    
    def __init__(self, vocab, max_size, embed_size=512, nhead=8, num_layers=2):
        super(TransformerModel, self).__init__()

        # each layer of the transformer
        encoder_layer = nn.TransformerEncoderLayer(embed_size, nhead)
        # build the transformer with n of the previous layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # embeds the input into a embed_size dimensional space
        self.src_emb = nn.Embedding(len(vocab), embed_size)
        # uses sine function to get positional embeddings
        self.pos_embed = PositionalEncoding(embed_size=embed_size, max_len=max_size)

        # vocab; includes stoi and itos look ups
        self.vocab = vocab
        self.embed_size = embed_size
        self.context_size = max_size
        
        # takes flattened output of last layer and maps to vocabulary size
        self.generator = Generator(embed_size*max_size, len(self.vocab))

    def forward(self, src):
        src = src.t()
        embed = self.pos_embed(self.src_emb(src) * math.sqrt(self.embed_size))
        output = self.encoder(embed).transpose(0,1)
        output = output.reshape(output.size()[0], -1)
        return self.generator(output)

    def predict_word(self, src):
        output = self.forward(src)
        return self.vocab.itos[torch.argmax(output)]

class Generator(nn.Module):
    
    # could change this to mean-pool or max pool
    def __init__(self, embed_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class PositionalEncoding(nn.Module):
    
    def __init__(self, embed_size=512, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

