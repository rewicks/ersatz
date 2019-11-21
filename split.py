import trainer
import model
import torch
import argparse

class EvalModel():
    def __init__(self, model_path):
        if torch.cuda.is_available():
            self.model = torch.load(model_path)   
        else:
            self.model = torch.load(model_path, map_location=torch.device('cpu'))
        if type(self.model) is torch.nn.DataParallel:
            self.model = self.model.module
        self.context_size = self.model.context_size//2

    def evaluate(self, file_path):
        output = []
        content = open(file_path, 'r').read().strip()
        content = content.split()
        for index, token in enumerate(content):
            temp_index = index-1
            left_context = []
            right_context = []
            if '\u2581' in token:
                while (len(left_context) < self.context_size):
                    if temp_index >= 0:
                        left_context.append(content[temp_index])
                    else:
                        left_context.append('<PAD>')
                    temp_index -= 1
                left_context.reverse()
                temp_index = index 
                while (len(right_context) < self.context_size):
                    if temp_index < len(content):
                        right_context.append(content[temp_index])
                    else:
                        right_context.append('<PAD>')
                    temp_index += 1
                context, _ = self.model.vocab.context_to_tensor([(left_context, right_context, '<HOLE>')])
                context = context.view(-1, 9)
                if self.model.predict_word(context) == '<eos>':
                    output.append('\n')
            output.append(token)
        output = ''.join(output)
        output = output.strip()
        return output

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--input')
    parser.add_argument('--output')

    args = parser.parse_args()
    model = EvalModel(args.model_path)
    output = model.evaluate(args.input)
    with open(args.output, 'w') as f:
        f.write(output) 
