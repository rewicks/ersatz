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

Batch = namedtuple("Batch", "contexts labels indices")

def load_model(checkpoint_path):
    model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = ErsatzTransformer(model_dict['vocab'], model_dict['left_context_size'], model_dict['right_context_size'], embed_size=model_dict['embed_size'], nhead=model_dict['nhead'], t_dropout=model_dict.get('t_dropout', 0.1), e_dropout=model_dict.get('e_dropout', 0.5), num_layers=2)
    model.load_state_dict(model_dict['weights'])
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
            self.vocab = self.model.vocab
            self.left_context_size = self.model.left_context_size
            self.right_context_size = self.model.right_context_size
            self.context_size = self.left_context_size + self.right_context_size + 1
            #self.spm_path = spm_path
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        else:
            self.model = load_model(model_path)
            #self.model = torch.load(model_path, map_location=torch.device('cpu'))
            if type(self.model) is torch.nn.DataParallel:
                self.model = self.model.module
            self.device = torch.device('cpu')
            self.model.to(self.device)
            self.vocab = self.model.vocab
            #self.context_size = self.model.context_size
            self.left_context_size = self.model.left_context_size
            self.right_context_size = self.model.right_context_size
            self.context_size = self.right_context_size + self.left_context_size + 1
            #self.spm_path = spm_path

    def evaluate(self, file_path):
        output = []
        content = open(file_path, 'r').read().strip()
        content = content.split()
        for index, token in enumerate(content):
            temp_index = index-1
            left_context = []
            right_context = []
            if '\u2581' in token:
                while (len(left_context) < self.left_context_size):
                    if temp_index >= 0:
                        left_context.append(content[temp_index])
                    else:
                        left_context.append('<PAD>')
                    temp_index -= 1
                left_context.reverse()
                temp_index = index 
                while (len(right_context) < self.right_context_size):
                    if temp_index < len(content):
                        right_context.append(content[temp_index])
                    else:
                        right_context.append('<PAD>')
                    temp_index += 1
                context, _ = self.vocab.context_to_tensor([(left_context, right_context, '<HOLE>')])
                context = context.view(-1, self.context_size)
                if self.model.module.predict_word(context) == '<eos>':
                    output.append('\n')
            output.append(token)
        output = ''.join(output)
        output = output.strip()
        return output

    def batchify(self, file_path, batch_size, det):
        for d in dataset.split_test_file(file_path, self.left_context_size, self.right_context_size):
            if len(d[0]) > 0:
                left_contexts = d[0]
                right_contexts = d[1]
                lines = []
                indices = []
                index = 0
                #import pdb; pdb.set_trace()
                for left, right in zip(left_contexts, right_contexts):
                    if det(detokenize(' '.join(left)), detokenize(' '.join(right))):
                        lines.append((left, right, '<eos>'))
                        indices.append(index)
                    index += 1 
                indices = torch.tensor(indices) 
                data, labels = self.vocab.context_to_tensor(lines)
         
                nbatch = data.size(0) // batch_size
                remainder = data.size(0) % batch_size 
                
                if remainder > 0:
                    remaining_data = data.narrow(0,nbatch*batch_size, remainder)
                    remaining_labels = labels.narrow(0, nbatch*batch_size, remainder)
                    remaining_indices = indices.narrow(0, nbatch*batch_size, remainder) 
                
                data = data.narrow(0, 0, nbatch*batch_size)
                labels = labels.narrow(0, 0, nbatch * batch_size)
                indices = indices.narrow(0, 0, nbatch * batch_size)

                data = data.view(batch_size, -1).t().contiguous()
                labels = labels.view(batch_size, -1).t().contiguous()
                indices = indices.view(batch_size, -1).t().contiguous()
                
                if remainder > 0:
                    remaining_data = remaining_data.view(remainder, -1).t().contiguous()
                    remaining_labels = remaining_labels.view(remainder, -1).t().contiguous()
                    remaining_indices = remaining_indices.view(remainder, -1).t().contiguous()
         

                batches = []
                data = data.view(-1, self.context_size, batch_size)
                labels = labels.view(-1, 1, batch_size)
                indices = indices.view(-1, 1, batch_size)

                if remainder > 0:
                    remaining_data = remaining_data.view(-1, self.context_size, remainder)
                    remaining_labels = remaining_labels.view(-1, 1, remainder)
                    remaining_indicies = remaining_indices.view(-1, 1, remainder)
                for context_batch, label_batch, index_batch in zip(data, labels, indices):
                    batches.append(Batch(context_batch.t(), label_batch[0], index_batch[0]))
                if remainder > 0:
                    batches.append(Batch(remaining_data[0].t(), remaining_labels[0][0], remaining_indices[0]))
                yield batches
            else:
                yield None

    def parallel_evaluation(self, input_path, batch_size, det=None):
        start = time.time()
        logging.info('starting batchify...')
        batches = self.batchify(input_path, batch_size, det)
        logging.info(f'batching took {time.time()-start} seconds')
        eos = []
        content = open(input_path).read().split('\n')
        for i, doc in enumerate(batches):
            if doc is not None:
                for batch in doc:
                    data = batch.contexts.to(self.device)
                    labels = batch.labels.to(self.device)
                    indices = torch.tensor(batch.indices)
            
                    output = self.model.forward(data)
                   
                    pred = output.argmax(1)
                    pred_ind = pred ^ labels
                    pred_ind = torch.where(pred_ind == 0)[0]
                    pred_ind = [indices[i].item() for i in pred_ind]
                    eos.extend(pred_ind)
                logging.info(f'finished making predictions...sorting...')
                if len(eos) == 0:
                    #print(content[i].strip())
                    yield content[i]
                else:
                    eos = sorted(eos)
                    next_index = int(eos.pop(0))
                    this_content = content[i].split()
                    output = ''
                    counter = 0
                    for index, word in enumerate(this_content):
                        #print(next_index,counter)
                        if counter == next_index:
                            output += '\n' + word
                            try:
                                next_index = int(eos.pop(0))
                            except:
                                next_index = -1
                        else:
                            output += ' ' + word
                        counter += 1
                    yield output
            else:
                yield ''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--batch_size', type=int)
    #parser.add_argument('--spm_path', default='/home/hltcoe/rwicks/ersatz/models/en.wiki.10000.model')

    args = parser.parse_args()
    model = EvalModel(args.model_path)
    model.model.eval()
    start = time.time()
    for output in model.parallel_evaluation(args.input, args.batch_size, det=Split()):
        output = output.split('\n')
        output = [o.strip() for o in output]
        print('\n'.join(output))
    elapsed = time.time()-start
    logging.info(f'{elapsed} seconds for: {args.input}')
    '''
    with open(args.output, 'w') as f:
        for line in output:
            f.write(line.strip() + '\n')
    '''
