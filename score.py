import argparse
import numpy as np
from determiner import *

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
        #print(f'Throwing out: {gold_sequence[:x]}\tdid not match {pred_sequence[:y]}')
        return gold_sequence[x:], pred_sequence[y:], levenshtein(gold_sequence[:x], pred_sequence[:y])
    else:
        exit(-1)


# def align(gold_sequence, pred_sequence):
#     distance = levenshtein(gold_sequence, pred_sequence)
#     x = 1
#     y = 1
#     min_edit = 0
#     while (distance[x][y] != min_edit) or gold_sequence[x] != pred_sequence[y] or gold_sequence in ['<mos>', '<eos>']:
#         min_edit = min (
#             distance[x][y+1],
#             distance[x+1][y+1],
#             distance[x+1][y]
#         )
#         if min_edit == distance[x][y+1]:
#             y += 1
#         elif min_edit == distance[x+1][y+1]:
#             x += 1
#             y += 1
#         else:
#             x += 1
#     print(f'Throwing out: {gold_sequence[:x]}\tdid not match {pred_sequence[:y]}')
#     return gold_sequence[x:], pred_sequence[y:], min_edit

# def original_context(content, context_index):
#     content = content.replace('\n', ' \u2581 ')
#     content = content.split()
#     i = 0
#     out = {}
#     for x, tok in enumerate(content):
#         for y, ch in enumerate(tok):
#             i += 1
#             if i == context_index:
#                 return ' '.join(content[x-4:x+1]).replace('\u2581', ' '), ' '.join(content[x+1:x+6]).replace('\u2581', ' ')
#             if y == len(tok)-1:
#                 if tok != '\u2581' and x < len(content) - 1 and content[x+1] != '\u2581':
#                     i += 1
#             else:
#                 i += 1

def make_context_mappings(content):
    content = content.replace('\n', ' \u2581 ')
    content = content.split()
    i = 0
    out = {}
    for x, tok in enumerate(content):
        for y, ch in enumerate(tok):
            i += 1
            out[i] = x+1
            # if i == context_index:
            #     return ' '.join(content[x-4:x+1]).replace('\u2581', ' '), ' '.join(content[x+1:x+6]).replace('\u2581', ' ')
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

def score(target_file, pred_file, det, rtl=False, range=20):
    pred_content = open(pred_file).read().strip()
    pred_gen = generator(pred_content)
    context_lookup, pred_content = make_context_mappings(pred_content)

    target_content = open(target_file).read().strip()
    target_gen = generator(target_content)

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

    while (index < len(pred_gen)):
        pred = pred_gen[index]
        target = target_gen[index]
        if (pred != target):
            if pred in ['<eos>', '<mos>'] and target in ['<eos>', '<mos>']:
                mapped_index = context_lookup[running_index]
                left_context = ' '.join(pred_content[mapped_index - 5:mapped_index])
                right_context = ' '.join(pred_content[mapped_index:mapped_index + 5]).replace('\u2581', ' ')

                if det(left_context, right_context):
                    print(f'{left_context} {pred} {right_context}')
                    if target == '<eos>':
                        incorrect_eos += 1
                    else:
                        incorrect_mos += 1
            elif pred in ['<eos>', '<mos>'] or target in ['<eos>', '<mos>']:
                exit(-1)
            else:
                rem_gold, rem_pred, edit_dist = align(target_gen[index:index+range], pred_gen[index:index+range])
                pred_gen = rem_pred + pred_gen[index+range:]
                target_gen = rem_gold + target_gen[index+range:]
                index = 0
                running_index += range-len(rem_pred)
                total_edits += edit_dist
        else:
            if pred in ['<eos>', '<mos>']:
                if pred_gen[index-1] in closing_punc or pred_gen[index-1] in ending_punc:
                    mapped_index = context_lookup[running_index]
                    left_context = ' '.join(pred_content[mapped_index - 5:mapped_index])
                    right_context = ' ' + ' '.join(pred_content[mapped_index:mapped_index + 5]).replace('\u2581', ' ')
                    if det(left_context, right_context):
                        if pred == '<eos>':
                            correct_eos += 1
                        elif pred == '<mos>':
                            correct_mos += 1

        index += 1
        running_index += 1
                  
    total = correct_eos + incorrect_eos + correct_mos + incorrect_mos
    accuracy = (correct_eos+correct_mos)/total
    recall = (correct_eos)/(correct_eos+incorrect_eos)
    try:
        precision = (correct_eos)/(correct_eos+incorrect_mos)
    except:
        precision = 'n/a'
    try:
        f1 = (2*precision*recall)/(precision+recall)
    except:
        f1 = 'n/a'
    print(f'Accuracy {accuracy*100:.2f}')
    print(f'Recall {recall*100:.2f}')
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rubric_file_path', type=str)
    parser.add_argument('pred_file_path', type=str)
    parser.add_argument('--rtl', action='store_true')
    parser.add_argument('--determiner_type', default='multilingual', choices=['en', 'multilingual', 'all'])
 
    args = parser.parse_args()

    if args.determiner_type == "en":
        determiner = PunctuationSpace()
    elif args.determiner_type == 'multilingual':
        determiner = MultilingualPunctuation()
    else:
        determiner = Split()

    score(args.rubric_file_path, args.pred_file_path, determiner, rtl=args.rtl)
