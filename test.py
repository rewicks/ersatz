import os, torch
import argparse
import hashlib
from utils import repackage_hidden
import time


'''
    Evalutation Model allows for testing of sentence segmentation
    with a pre-trained model. Please see train.py for training
    your own model

'''
class EvalModel():
    
    def __init__(self, model_path, corpus_path, tokenizer):
        self.model, _, _ = self.load_model(model_path)
        self.dictionary = self.model.dictionary
        self.tokenzier = tokenizer
        self.context = ''   # keeps track of the words that have thus far been given as input to the model

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            return torch.load(f, map_location=torch.device('cpu'))

    def load_corpus(self, test_data):
        fn = 'corpus.{}.data'.format(hashlib.md5(corpus_path.encode()).hexdigest())
        return torch.load(fn).dictionary    

    # currently a redundant function to get the input in the correct
    # dimensions for the model
    def batchify(self, data):
        data = data.narrow(0, 0, data.size(0))
        data = data.view(1, -1).t().contiguous()
        return data

    # resets the model, and gives it the next word to initialize
    # the hidden states
    def reset_model(self, context_word):
        self.model.reset()
        self.step(context_word, reset=True)

    # feeds the next word into the model, saves the outputs
    def step(self, context_word, reset=False):
        if reset:
            self.context = ''
            self.hidden = self.model.init_hidden(1)
        else:
            self.hidden = repackage_hidden(self.hidden)
        
        self.context += context_word + ' '
        inputs = torch.LongTensor(1)
        inputs[0] = self.dictionary.word2idx[context_word]
        inputs = self.batchify(inputs)
        
        self.output, self.hidden = self.model(inputs, self.hidden)

    def evaluate(self, test_data):
        self.model.eval()
        
        results = {}

        # iterates over a test file
        # must be in the format of one sentence per line
        # with no extra tags
        def next_token():
            with open(test_data, 'r') as f:
                for line in f:
                    words = self.tokenizer(line) + ['<eos>']
                    for w in words:
                        yield w
        
        all_tokens = next_token()
        # initializes the model with the first word of the file
        # note that the model will never predict the first word of the file
        self.reset_model(next(all_tokens))

        counter = 1
        start = time.time()
        for observed_word in all_tokens:
            predicted_word = self.dictionary.idx2word[self.model.decoder(self.output).softmax(1).argmax()]
            
            if predicted_word == '<eos>':
                self.reset_model(observed_word)
            elif observed_word != '<eos>':
                self.step(observed_word)

            #print(f'{counter} at output: {predicted_word} vs actual: {obs}')
            print(f'{counter}: {self.context} [{observed_word}]: {predicted_word}')

            counter += 1
            
            # dictionary is a confusion matrix
            # first key is the observed_word
            
            if observed_word in results:
                results[observed_word][predicted_word] = 1 + results[observed_word].get(predicted_word, 0)
            else:
                results[observed_word] = {}
                results[observed_word][predicted_word] = 1
 
        return results

    def compile_results(self, results):
        matrix = [[0,0][0,0]]
        for obs in results:
            if obs == '<eos>':
                for pred in results[obs]:
                    if pred == '<eos>':
                        matrix[0][0] += 1
                    else:
                        matrix[0][1] += 1
            else:
                for pred in results[obs]:
                    if pred == '<eos>':
                        matrix[1][0] += 1
                    else:
                        matrix[1][1] += 1
        return results
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_path')
    parser.add_argument('corpus_path')
    parser.add_argument('test_data_path')

    args = parser.parse_args()

    evaluator = EvalModel(args.model_path, args.corpus_path)
    results = evaluator.evaluate(args.test_data_path) 
    print(results)
    print(evaluator.compile_results(results))
