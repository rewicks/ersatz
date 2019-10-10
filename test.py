import os, torch
import argparse
import hashlib
from utils import repackage_hidden
import time
import Tokenizer
import glob
from pathlib import Path

'''
    Evalutation Model allows for testing of sentence segmentation
    with a pre-trained model. Please see train.py for training
    your own model

'''

LOGS = '/home/hltcoe/rwicks/ersatz/testing-logs/'

class EvalModel():
    
    def __init__(self, model_path, tokenizer):
        self.model, _, _ = self.load_model(model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()    
    
        self.dictionary = self.model.dictionary
        self.tokenizer = Tokenizer.SPM(tokenizer)
        self.context = ''   # keeps track of the words that have thus far been given as input to the model
        affix = model_path.split('/')[-1].split('.')[0] + '.txt'
        self.log = open(os.path.join(LOGS, affix), 'w')

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            return torch.load(f, map_location=torch.device('cpu'))

    # currently a redundant function to get the input in the correct
    # dimensions for the model
    def batchify(self, data):
        data = data.narrow(0, 0, data.size(0))
        data = data.view(1, -1).t().contiguous()
        if torch.cuda.is_available():
            data = data.cuda()
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
        
        output_name = test_data.split('/')[-1].split('.')
        output_name = 'segmented/' +  ''.join(output_name[0:-1]) + '.seg.' + output_name[-1]

        output = open(output_name, 'w')
        # iterates over a test file
        # must be in the format of one sentence per line
        # with no extra tags
    
        def next_token():
            with open(test_data, 'r') as f:
                for line in f:
                    words = self.tokenizer.encode(line)
                    for w in words:
                        yield w
        
        all_tokens = next_token()
        # initializes the model with the first word of the file
        # note that the model will never predict the first word of the file
        first_word = next(all_tokens)
        self.reset_model(first_word)
        line = first_word

        counter = 1
        start = time.time()
        for observed_word in all_tokens:
            predicted_word = self.dictionary.idx2word[self.model.decoder(self.output).softmax(1).argmax()]
            
            self.log.write(f'{counter}: {self.context} [{observed_word}]: {predicted_word}\n')

            if predicted_word == '<eos>':
                output.write(line + '\n')
                line = observed_word
                self.reset_model(observed_word)
            elif observed_word != '<eos>':
                line += ' ' + observed_word
                self.step(observed_word)

            counter += 1
            
            # dictionary is a confusion matrix
            # first key is the observed_word
            
            #if observed_word in results:
            #    results[observed_word][predicted_word] = 1 + results[observed_word].get(predicted_word, 0)
            #else:
            #    results[observed_word] = {}
            #    results[observed_word][predicted_word] = 1
 
        #return results
        output.close()


def compile_results(results):
    matrix = [[0,0],[0,0]]
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
    return matrix
   

# Adds the results of one test file to another 
def combine(results_one, results_two):
    for obs in results_two:
        for pred in results_two[obs]:
            if obs in results_one:
                if pred in results_one[obs]:
                    results_one[obs][pred] += results_two[obs][pred]
    return results_one

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('language_model_path')
    parser.add_argument('tokenizer_path')
    parser.add_argument('test_files', type=Path, nargs='*')

    args = parser.parse_args()

    evaluator = EvalModel(args.language_model_path, args.tokenizer_path, args.output)
    
    results = {}
    
    #test_files = glob.glob(args.test_files)
    #print(len(test_files))
    for test_data_path in args.test_files:
        print(f'Testing on {test_data_path}')
        print('-'*100) 
        these_results = evaluator.evaluate(test_data_path)
        results = combine(these_results, results) 
        #print(f'Results on this file:\n{these_results}')
    #print(f'Current cummulative results {results}')
    print(f'Current compiled results {compile_results(results)}')
