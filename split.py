from model import ErsatzTransformer
import torch
import argparse
import dataset
import sys
from determiner import *

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
    def __init__(self, model_path, args):
        if torch.cuda.is_available() and not args.cpu:
            print('Using gpu', file=sys.stderr)
            self.model = load_model(model_path)
            if type(self.model) is torch.nn.DataParallel:
                self.model = self.model.module
            self.device = torch.device("cuda") 
            self.model = self.model.to(self.device)  
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using cpu', file=sys.stderr)
            self.model = load_model(model_path)
            if type(self.model) is torch.nn.DataParallel:
                self.model = self.model.module
            self.device = torch.device('cpu')
            self.model.to(self.device)

        self.tokenizer = self.model.tokenizer
        self.left_context_size = self.model.left_context_size
        self.right_context_size = self.model.right_context_size
        self.context_size = self.right_context_size + self.left_context_size

    def batchify(self, content, batch_size, det):
        left_contexts, right_contexts = dataset.split_test_file(content, self.tokenizer, self.left_context_size, self.right_context_size)
        if len(left_contexts) > 0:
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
            return batches
        else:
            return []

    def parallel_evaluation(self, content, batch_size, det=None, min_sent_length=3):
        batches = self.batchify(content, batch_size, det)
        eos = []
        for contexts, indices, in batches:
            data = contexts.to(self.device)

            output = self.model.forward(data)

            pred = output.argmax(1)
            pred_ind = torch.where(pred == 0)[0]
            pred_ind = [indices[i].item() for i in pred_ind]
            eos.extend(pred_ind)
        if len(eos) == 0:
            yield content.strip()
        else:
            eos = sorted(eos)
            next_index = int(eos.pop(0))
            this_content = self.tokenizer.encode(content, out_type=str)
            output = []
            counter = 0
            for index, word in enumerate(this_content):
                if counter == next_index:
                    try:
                        next_index = int(eos.pop(0))
                    except:
                        next_index = len(content)-1
                    if (next_index - counter >= 5):
                        output.append('\n')
                    output.append(word)
                    
                else:
                    output.append(word)
                counter += 1
            output = self.tokenizer.merge(output, technique='utility').strip().split('\n')
            yield '\n'.join([o.strip() for o in output])
        yield ''

    def split(self, input_file, output_file, batch_size, det=None):
        for line in input_file:
            for batch_output in self.parallel_evaluation(line, batch_size, det=det):
                print(batch_output.strip(), file=output_file)

    def split_delimiter(self, input_file, output_file, batch_size, delimiter, text_ids, det=None):
        for line in input_file:
            line = line.split(delimiter)
            new_lines = []
            max_len = 1
            for i, l in enumerate(line):
                if i in text_ids:
                    for batch_output in self.parallel_evaluation(l, batch_size, det=det):
                        batch_output = batch_output.split('\n')
                        new_lines.append(batch_output)
                        if len(batch_output) > max_len:
                            max_len = len(batch_output)
                else:
                    new_lines.append([line[i]])
            for x in range(max_len):
                out_line = []
                for i, col in enumerate(new_lines):
                    if x >= len(col):
                        if i not in text_ids:
                            out_line.append(col[-1])
                        else:
                            out_line.append('')
                    else:
                        out_line.append(col[x])
                print(delimiter.join(out_line).strip(), file=output_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--input', default=None, help="Input file. None means stdin")
    parser.add_argument('--output', default=None, help="Output file. None means stdout")
    parser.add_argument('--delim', type=str, default='\t', help="Delimiter character. Default is tab")
    parser.add_argument('--text_ids', type=str, default=None, help="Columns to split (comma-separated; 0-indexed). If empty, plain-text")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size--predictions to make at once")
    parser.add_argument('--determiner_type', default='multilingual', choices=['en', 'multilingual', 'all'])
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    if args.determiner_type == "en":
        determiner = PunctuationSpace()
    elif args.determiner_type == 'multilingual':
        determiner = MultilingualPunctuation()
    else:
        determiner = Split()

    if args.input is not None:
        input_file = open(args.input, 'r')
    else:
        input_file = sys.stdin

    if args.output is not None:
        output_file = open(args.output, 'w')
    else:
        output_file = sys.stdout

    model = EvalModel(args.model_path, args)

    with torch.no_grad():
        if args.text_ids is None:
            model.split(input_file, output_file, args.batch_size, det=determiner)
        else:
            text_ids = [int(t) for t in args.text_ids.split(',')]
            model.split_delimiter(input_file, output_file, args.batch_size, args.delim, text_ids, det=determiner)
    print('complete', file=sys.stderr)
