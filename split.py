import trainer
#from trainer import ErsatzTrainer
import model
import torch
import argparse
import dataset
from collections import namedtuple
import time
import logging

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)

Batch = namedtuple("Batch", "contexts labels indices")

class EvalModel():
    def __init__(self, model_path, spm_path):
        if torch.cuda.is_available():
            self.model = torch.load(model_path)
            if type(self.model) is torch.nn.DataParallel:
                self.model = self.model.module
            self.device = torch.device("cuda") 
            self.model = self.model.to(self.device)  
            self.vocab = self.model.vocab
            self.context_size = self.model.context_size
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        else:
            self.model = torch.load(model_path, map_location=torch.device('cpu'))
            if type(self.model) is torch.nn.DataParallel:
                self.model = self.model.module
            self.device = torch.device('cpu')
            self.model.to(self.device)
            self.vocab = self.model.vocab
            #self.context_size = self.model.context_size
            self.left_context_size = self.model.left_context_size
            self.right_context_size = self.model.right_context_size
            self.context_size = self.right_context_size + self.left_context_size + 1
            self.spm_path = spm_path

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

    def batchify(self, file_path, batch_size):
        left_contexts, right_contexts = dataset.split_test_file(file_path, self.left_context_size, self.right_context_size, spm_path=self.spm_path)
        lines = []
        for left, right in zip(left_contexts, right_contexts):
            lines.append((left, right, '<eos>'))        
        indices = [x for x in range(len(lines))]
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
        return batches

    def parallel_evaluation(self, input_path, batch_size):
        batches = self.batchify(input_path, batch_size)
        eos = []
        for batch in batches:
            data = batch.contexts.to(self.device)
            labels = batch.labels.to(self.device)
            indices = torch.tensor(batch.indices)
    
            output = self.model.forward(data)
           
            pred = output.argmax(1)
            pred_ind = pred ^ labels
            pred_ind = torch.where(pred_ind == 0)[0]
            pred_ind = [indices[i].item() for i in pred_ind]
            eos.extend(pred_ind)
        eos = sorted(eos)
        next_index = int(eos.pop(0))
        content = open(input_path).read()
        content = content.split()
        output = ''
        counter = 0
        for index, word in enumerate(content):
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
        return output 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--spm_path', default='/home/hltcoe/rwicks/ersatz/models/en.wiki.10000.model')

    args = parser.parse_args()
    model = EvalModel(args.model_path, args.spm_path)
    model.model.eval()
    start = time.time()
    output = model.parallel_evaluation(args.input, args.batch_size)
    elapsed = time.time()-start
    logging.info(f'{elapsed} seconds for: {args.input}')
    output = output.split('\n')
    with open(args.output, 'w') as f:
        for line in output:
            f.write(line.strip() + '\n')
