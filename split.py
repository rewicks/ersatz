import trainer
#from trainer import ErsatzTrainer
from model import ErsatzTransformer
import torch
import argparse
import dataset
from collections import namedtuple
import time
import logging
from determiner import *

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)


# default args for loading models
# should write something to merge default args with loaded args (overwrite when applicable)
class DefaultArgs():
    def __init__(self):
        self.left_context_size=4
        self.right_context_size=6
        self.embed_size=256
        self.nhead=8
        self.dropout=0.1
        self.transformer_nlayers=2    
        self.linear_nlayers=0
        self.activation_type='tanh'


def load_model(checkpoint_path):
    model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = ErsatzTransformer(model_dict['tokenizer'], model_dict['args'])
    model.load_state_dict(model_dict['weights'])
    model.eval()
    return model

def detokenize(input_string):
    input_string = input_string.replace(' ', '')
    input_string = input_string.replace('\u2581', ' ')
    return input_string

class EvalModel():
    def __init__(self, model_path):
        if torch.cuda.is_available():
            self.model = load_model(model_path)
            if type(self.model) is torch.nn.DataParallel:
                self.model = self.model.module
            self.device = torch.device("cuda") 
            self.model = self.model.to(self.device)  
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        else:
            self.model = load_model(model_path)
            if type(self.model) is torch.nn.DataParallel:
                self.model = self.model.module
            self.device = torch.device('cpu')
            self.model.to(self.device)

        self.tokenizer = self.model.tokenizer
        self.left_context_size = self.model.left_context_size
        self.right_context_size = self.model.right_context_size
        self.context_size = self.right_context_size + self.left_context_size

    def batchify(self, file_path, batch_size, det):
        for d in dataset.split_test_file(file_path, self.tokenizer, self.left_context_size, self.right_context_size):
            if len(d[0]) > 0:
                left_contexts = d[0]
                right_contexts = d[1]
                lines = []
                indices = []
                index = 0
                for left, right in zip(left_contexts, right_contexts):
                    if det(detokenize(' '.join(left)), detokenize(' '.join(right))):
                        lines.append((left, right, '<eos>'))
                        indices.append(index)
                    index += 1 
                indices = torch.tensor(indices) 
                data, _ = self.tokenizer.context_to_tensor(lines)
         
                nbatch = data.size(0) // batch_size
                remainder = data.size(0) % batch_size 
                
                if remainder > 0:
                    remaining_data = data.narrow(0,nbatch*batch_size, remainder)
                    remaining_indices = indices.narrow(0, nbatch*batch_size, remainder)
                
                data = data.narrow(0, 0, nbatch*batch_size)
                indices = indices.narrow(0, 0, nbatch * batch_size)

                data = data.view(batch_size, -1).t().contiguous()
                indices = indices.view(batch_size, -1).t().contiguous()
                
                if remainder > 0:
                    remaining_data = remaining_data.view(remainder, -1).t().contiguous()
                    remaining_indices = remaining_indices.view(remainder, -1).t().contiguous()
         

                batches = []
                data = data.view(-1, self.context_size, batch_size)
                indices = indices.view(-1, 1, batch_size)

                if remainder > 0:
                    remaining_data = remaining_data.view(-1, self.context_size, remainder)
                    remaining_indices = remaining_indices.view(-1, 1, remainder)
                for context_batch, index_batch in zip(data, indices):
                    batches.append((context_batch.t(), index_batch[0]))
                if remainder > 0:
                    batches.append((remaining_data[0].t(), remaining_indices[0][0]))
                yield batches
            else:
                yield None

    def parallel_evaluation(self, input_path, batch_size, det=None):
        batches = self.batchify(input_path, batch_size, det)
        eos = []
        content = open(input_path).read().split('\n')
        for i, doc in enumerate(batches):
            if doc is not None:
                for contexts, indices in doc:
                    data = contexts.to(self.device)
            
                    output = self.model.forward(data)
                   
                    pred = output.argmax(1)
                    pred_ind = torch.where(pred == 0)[0]
                    pred_ind = [indices[i].item() for i in pred_ind]
                    eos.extend(pred_ind)
                if len(eos) == 0:
                    yield content[i]
                else:
                    eos = sorted(eos)
                    next_index = int(eos.pop(0))
                    this_content = self.tokenizer.encode(content[i], out_type=str)
                    output = []
                    counter = 0
                    for index, word in enumerate(this_content):
                        if counter == next_index:
                            output.append('\n')
                            output.append(word)
                            try:
                                next_index = int(eos.pop(0))
                            except:
                                next_index = -1
                        else:
                            output.append(word)
                        counter += 1
                    output = self.tokenizer.merge(output, technique='utility')
                    yield output
            else:
                yield ''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--determiner_type', default='en', choices=['en', 'multilingual', 'all'])

    args = parser.parse_args()

    if args.determiner_type == "en":
        determiner = PunctuationSpace()
    elif args.determiner_type == 'multilingual':
        determiner = MultilingualPunctuation()
    else:
        determiner = Split()

    model = EvalModel(args.model_path)
    for output in model.parallel_evaluation(args.input, args.batch_size, det=determiner):
        output = output.split('\n')
        output = [o.strip() for o in output]
        print('\n'.join(output))
