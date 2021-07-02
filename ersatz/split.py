#!/usr/bin/env python

import pathlib
import os
import torch
import argparse
import sys
import csv

if __package__ is None and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'ersatz'

from . import __version__
from .utils import get_model_path, list_models, MODELS
from .model import ErsatzTransformer
from .dataset import SourceFactors, split_test_file
from .candidates import PunctuationSpace, MultilingualPunctuation, Split
from .subword import SentencePiece

import logging


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("ersatz")

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
    tokenizer = SentencePiece(serialization=model_dict['tokenizer'])
    model = ErsatzTransformer(tokenizer, model_dict['args'])
    model.load_state_dict(model_dict['weights'])
    model.eval()
    return model

def detokenize(input_string):
    input_string = input_string.replace(' ', '')
    input_string = input_string.replace('\u2581', ' ')
    return input_string

class EvalModel():
    def __init__(self, model_path):
        self.model = load_model(model_path)
        if type(self.model) is torch.nn.DataParallel:
            self.model = self.model.module

        self.tokenizer = self.model.tokenizer
        self.left_context_size = self.model.left_context_size
        self.right_context_size = self.model.right_context_size
        self.context_size = self.right_context_size + self.left_context_size

    def batchify(self, content, batch_size, candidates):
        source_factors = SourceFactors()
        left_contexts, right_contexts = split_test_file(content, self.tokenizer, self.left_context_size, self.right_context_size)
        if len(left_contexts) > 0:
            lines = []
            indices = []
            index = 1
            for left, right in zip(left_contexts, right_contexts):
                if candidates(detokenize(' '.join(left)), detokenize(' '.join(right))):
                    lines.append((left, source_factors.compute(left),
                                  right, source_factors.compute(right),
                                  '<eos>'))
                    indices.append(index)
                index += 1
            indices = torch.tensor(indices)
            data, factors, _ = self.tokenizer.context_to_tensor(lines)

            nbatch = data.size(0) // batch_size
            remainder = data.size(0) % batch_size

            if remainder > 0:
                remaining_data = data.narrow(0, nbatch*batch_size, remainder)
                remaining_factors = factors.narrow(0, nbatch*batch_size, remainder)
                remaining_indices = indices.narrow(0, nbatch*batch_size, remainder)

            data = data.narrow(0, 0, nbatch*batch_size)
            factors = factors.narrow(0, 0, nbatch*batch_size)
            indices = indices.narrow(0, 0, nbatch * batch_size)

            data = data.view(batch_size, -1).t().contiguous()
            factors = factors.view(batch_size, -1).t().contiguous()
            indices = indices.view(batch_size, -1).t().contiguous()

            if remainder > 0:
                remaining_data = remaining_data.view(remainder, -1).t().contiguous()
                remaining_factors = remaining_factors.view(remainder, -1).t().contiguous()
                remaining_indices = remaining_indices.view(remainder, -1).t().contiguous()


            batches = []
            data = data.view(-1, self.context_size, batch_size)
            factors = factors.view(-1, self.context_size, batch_size)
            indices = indices.view(-1, 1, batch_size)

            if remainder > 0:
                remaining_data = remaining_data.view(-1, self.context_size, remainder)
                remaining_factors = remaining_factors.view(-1, self.context_size, remainder)
                remaining_indices = remaining_indices.view(-1, 1, remainder)
            for context_batch, factors_batch, index_batch in zip(data, factors, indices):
                batches.append((context_batch.t(), factors_batch.t(), index_batch[0]))
            if remainder > 0:
                batches.append((remaining_data[0].t(), remaining_factors[0].t(), remaining_indices[0][0]))
            return batches
        else:
            return []

    def parallel_evaluation(self, content, batch_size, candidates=None, min_sent_length=3):
        batches = self.batchify(content, batch_size, candidates)
        eos = []
        for contexts, factors, indices, in batches:
            data = contexts.to(self.device)
            if not self.model.source_factors:
                factors = None
            else:
                factors = factors.to(self.device)

            output = self.model.forward(data, factors=factors)

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
        yield None

    def split(self, input_file, output_file, batch_size, candidates=None):
        for line in input_file:
            for batch_output in self.parallel_evaluation(line, batch_size, candidates=candidates):
                if batch_output is not None:
                    print(batch_output.strip(), file=output_file)
        return output_file


    def split_delimiter(self, input_file, output_file, batch_size, delimiter, columns, candidates=None):
        input_file = csv.reader(input_file, delimiter=delimiter)
        for line in input_file:
            new_lines = []
            max_len = 1
            for i, l in enumerate(line):
                if i in columns:
                    for batch_output in self.parallel_evaluation(l, batch_size, candidates=candidates):
                        if batch_output is not None:
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
                        if i not in columns:
                            out_line.append(col[-1])
                        else:
                            out_line.append('')
                    else:
                        out_line.append(col[x])
                print(delimiter.join(out_line).strip(), file=output_file)

def parse_args():
    parser = argparse.ArgumentParser(
        description="ERSATZ SEGMENTER: Segments input text into sentences.\n"
        "      Example: ersatz --model fr --input wikipedia.fr --output output.fr",
        usage='%(prog)s [-h] [--model MODEL] [--input INPUT] [--output OUTPUT] [OPTIONS]',
        formatter_class=argparse.RawTextHelpFormatter
    )

    main_group = parser.add_argument_group('arguments')
    main_group.add_argument('--model', '-m', default='default-multilingual',
                        help="Either name of or path to a pre-trained ersatz model")
    main_group.add_argument('--input', '-i', default=None,
                        help="Input file. None means stdin")
    main_group.add_argument('--output', '-o', default=None,
                        help="Output file. None means stdout")
    main_group.add_argument('--batch-size', '-b', type=int, default=16,
                        help="Batch size--predictions to make at once")
    main_group.add_argument('--candidates', '-c', default='multilingual', choices=['multilingual', 'en', 'all'],
                        help = "Criteria for selecting candidate sites. Defaults to 'multilingual'\n"
                               "   * multilingual: [EOS punctuation][!number] (sentence-ending punctuation followed by a non-digit)\n"
                               "   * en: [EOS punctuation][any_punctuation]*[space] (sentence-ending punctuation followed by a space)\n"
                               "   * all: all possible contexts")
    main_group.add_argument('--cpu', action='store_true', help="Uses CPU (GPU is default if available)")

    tsv_group = parser.add_argument_group('tsv options', description="Used for splitting .csv/.tsv/etc files. This mode triggered by '--columns'")
    tsv_group.add_argument('--delimiter', '-d', type=str, default='\t',
                        help="Delimiter character (default is \\t)\n"
                             "  * '--columns' must be set"
                        )
    tsv_group.add_argument('--columns', '-C', type=int, default=None, nargs="*",
                        help="Columns to split (0-indexed). If empty, plain-text\n"
                        )

    options = parser.add_argument_group('additional options')
    options.add_argument('--version', '-V', action='store_true', help="Prints ersatz version")
    options.add_argument('--download', '-D', action='store_true',
                        help="Downloads model selected via '--model'")
    options.add_argument('--list', '-l', action='store_true',
                        help="Lists available models.")
    options.add_argument('--quiet', '-q', action='store_true',
                        help="Disables logging.")

    args = parser.parse_args()

    args.text = None

    return args

def split(args):
    if args.candidates == "en":
        candidates = PunctuationSpace()
    elif args.candidates == 'multilingual':
        candidates = MultilingualPunctuation()
    else:
        candidates = Split()

    if args.input is not None:
        input_file = open(args.input, 'r')
    elif args.text is not None:
        input_file = args.text.split('\n')
    else:
        input_file = sys.stdin

    if args.output is not None:
        output_file = open(args.output, 'w')
    elif args.text is not None:
        from io import StringIO
        output_file = StringIO()
    else:
        output_file = sys.stdout

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.model not in MODELS:
        model = EvalModel(args.model)
    else:
        model_path = get_model_path(args.model)
        model = EvalModel(model_path)

    model.model = model.model.to(device)
    model.device = device

    with torch.no_grad():
        if args.columns is None:
            output_file = model.split(input_file, output_file, args.batch_size, candidates=candidates)
        else:
            output_file = model.split_delimiter(input_file, output_file, args.batch_size, args.delimiter, args.columns, candidates=candidates)

    if args.text:
        return output_file.getvalue().strip().split('\n')

def main():

    args = parse_args()

    if args.version:
        from . import __version__
        print("ersatz", __version__)
        sys.exit(0)

    if args.download:
        get_model_path(args.model)
        sys.exit(0)

    if args.list:
        list_models()
        sys.exit(0)

    if args.quiet:
        logger.setLevel(logging.ERROR)

    split(args)

if __name__ == '__main__':
    main()
