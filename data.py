import os
import torch
import Tokenizer
from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class TrainingCorpus(object):
    def __init__(self, training_path, tokenizer_path):
        self.dictionary = Dictionary()
        self.tokenizer = Tokenizer.SPM(tokenizer_path)
        self.train, self.valid = self.tokenize(training_path)
        

    def tokenize(self, path, train_percent=0.5):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        train_percent = round(train_percent, 2)

        # Get length of file
        sentences = 0
        with open(path, 'r') as f:
            for line in f:
                sentences += 1

        training_sentences = int(sentences*train_percent)
        # Add words to the dictionary
        counter = 0
        training_tokens = 0
        validation_tokens = 0
        with open(path, 'r') as f:
            for line in f:
                words = self.tokenizer.encode(line) + ['<eos>']
                if counter <= training_sentences:
                    training_tokens += len(words)
                else:
                    validation_tokens += len(words)
                counter += 1
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            training_ids = torch.LongTensor(training_tokens)
            validation_ids = torch.LongTensor(validation_tokens)

            token = 0
            counter = 0
            for line in f:
                words = self.tokenizer.encode(line) + ['<eos>']
                for word in words:
                    if counter <= training_sentences:
                        training_ids[token] = self.dictionary.word2idx[word]
                    else:
                        validation_ids[token] = self.dictionary.word2idx[word]
                    token += 1
                    if counter == training_sentences:
                        tokens = 0
                    counter += 1

        return training_ids, validation_ids

