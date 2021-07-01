import argparse
import numpy as np
import pathlib
import sys

if __package__ is None and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'ersatz'

from .candidates import MultilingualPunctuation, PunctuationSpace, Split

def levenshtein(gold_sequence, pred_sequence):
    size_x = len(gold_sequence) + 1
    size_y = len(pred_sequence) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if gold_sequence[x-1] not in ['<eos>', '<mos>'] and gold_sequence[x-1] == pred_sequence[y-1]:
                matrix[x][y] = min (
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1],
                    matrix[x][y-1] + 1
                )
            else:
                matrix[x][y] = min (
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1] + 1,
                    matrix[x][y-1] + 1
                )
    return matrix[size_x-1][size_y-1]

def subset(one, two):
    for a, b in zip(one, two):
        if a != b:
            if a in ['<eos>', '<mos>'] and b not in ['<eos>', '<mos>']:
                return False
    return True

def align(gold_sequence, pred_sequence):
    x = 0
    y = 0
    while gold_sequence[x] not in pred_sequence:
        x += 2 
    while pred_sequence[y] != gold_sequence[x]:
        y += 2
    if subset(gold_sequence[x:], pred_sequence[y:]):
        return gold_sequence[x:], pred_sequence[y:], levenshtein(gold_sequence[:x], pred_sequence[:y])
    else:
        gold_sequence_prefix, pred_sequence_prefix, min_edit = align(gold_sequence[:-2], pred_sequence[:-2])
        return gold_sequence_prefix + gold_sequence[-2:], pred_sequence_prefix + pred_sequence[-2:], min_edit

def make_context_mappings(content):
    content = content.replace('\n', ' \u2581 ')
    content = content.split()
    i = 0
    out = {}
    for x, tok in enumerate(content):
        for y, ch in enumerate(tok):
            i += 1
            out[i] = (x+1,y)
            if y == len(tok)-1:
                if tok != '\u2581' and x < len(content) - 1 and content[x+1] != '\u2581':
                    i += 1
            else:
                i += 1
    return out, content



def generator(content):
    out = []
    content = content.replace('\n', '\u2581')
    content = ''.join(content.split())
    for i,c in enumerate(content):
        if c == '\u2581':
            out.append('<eos>')
        else:
            out.append(c)
            if i < len(content)-1 and content[i+1] != '\u2581':
                out.append('<mos>')
    return out


def score(target_file, pred_file):
    pred_content = open(pred_file).read().strip()
    pred_gen = generator(pred_content)
    #context_lookup, pred_content = make_context_mappings(pred_content)

    target_content = open(target_file).read().strip()
    target_gen = generator(target_content)
    context_lookup, target_content = make_context_mappings(target_content)
    
    correct_eos = 0
    incorrect_eos = 0
    correct_mos = 0
    incorrect_mos = 0
    index = 0
    running_index = 0
    total_edits = 0
    errors = []
    
    type_one = []
    type_two = []
    reached = False
    while (index < len(pred_gen)):
        pred = pred_gen[index]
        target = target_gen[index]
        if (pred != target):
            if pred in ['<eos>', '<mos>'] and target in ['<eos>', '<mos>']:
                mapped_index_x, mapped_index_y = context_lookup[running_index]
                left_context = target_content[mapped_index_x-5:mapped_index_x-1]
                right_context = target_content[mapped_index_x:mapped_index_x+5]

                
                if mapped_index_x < len(target_content) and mapped_index_y == len(target_content[mapped_index_x])-1:
                    left_context += [target_content[mapped_index_x-1]]
                else:
                    left_context += [target_content[mapped_index_x-1][:mapped_index_y+1]]
                    right_context = [target_content[mapped_index_x-1][mapped_index_y+1:]] + right_context

                left_context = ' '.join(left_context)
                right_context = ' '.join(right_context).replace('\u2581', ' ')
                print(f'{left_context} {pred} {right_context}')
                if target == '<eos>':
                    incorrect_eos += 1
                else:
                    incorrect_mos += 1

            elif pred in ['<eos>', '<mos>'] or target in ['<eos>', '<mos>']:
                exit(-1)
            else:
                SUCCESS = False
                range = 4
                while not SUCCESS:
                    try:
                        rem_gold, rem_pred, edit_dist = align(target_gen[index:index+range], pred_gen[index:index+range])
                        running_index += range-len(rem_gold)
                        pred_gen = rem_pred + pred_gen[index+range:]
                        target_gen = rem_gold + target_gen[index+range:]
                        SUCCESS = True
                    except:
                        # expands window to align in until match is found
                        range += 2
                index = 0
                total_edits += edit_dist
        else:
            if pred in ['<eos>', '<mos>']:
                if pred == '<eos>':
                    correct_eos += 1
                else:
                    correct_mos += 1
        index += 1
        running_index += 1
                  
    total = correct_eos + incorrect_eos + correct_mos + incorrect_mos
    try:
        accuracy = (correct_eos+correct_mos)/total
    except:
        accuracy  = 'n/a'
    try:
        recall = (correct_eos)/(correct_eos+incorrect_eos)
    except:
        recall = 'n/a'
    try:
        precision = (correct_eos)/(correct_eos+incorrect_mos)
    except:
        precision = 'n/a'
    try:
        f1 = (2*precision*recall)/(precision+recall)
    except:
        f1 = 'n/a'
    try:
        print(f'Accuracy {accuracy*100:.2f}')
    except:
        print(f'Accuracy n/a')
    try:
        print(f'Recall {recall*100:.2f}')
    except:
        print(f'Recall n/a')
    try:
        print(f'Precision {precision*100:.2f}')
    except:
        print("Precision n/a")
    try:
        print(f'F1 {f1*100:.2f}')
    except:
        print("F1 n/a")
    for one in type_one:
        print(one)
    for two in type_two:
        print(two)
    print(total_edits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rubric_file_path', type=str)
    parser.add_argument('pred_file_path', type=str)
    parser.add_argument('--determiner_type', default='multilingual', choices=['en', 'multilingual', 'all'])
 
    args = parser.parse_args()

    if args.determiner_type == "en":
        determiner = PunctuationSpace()
    elif args.determiner_type == 'multilingual':
        determiner = MultilingualPunctuation()
    else:
        determiner = Split()

    score(args.rubric_file_path, args.pred_file_path)

if __name__ == '__main__':
    main()
