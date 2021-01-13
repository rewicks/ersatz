import argparse
from determiner import *

def score(target_file, pred_file, det, rtl=False):
    pred_content = open(pred_file).read()
    pred_content = pred_content.replace(' ', ' <mos> ')
    pred_content = pred_content.replace('\n', ' <eos> ')
    
    target_content = open(target_file).read()
    target_content = target_content.replace(' ', ' <mos> ')
    target_content = target_content.replace('\n', ' <eos> ')

    pred_content = pred_content.split()
    target_content = target_content.split()
    
    correct_eos = 0
    incorrect_eos = 0
    correct_mos = 0
    incorrect_mos = 0
    index = 0
    
    type_one = []
    type_two = []

    for pred, target in zip(pred_content, target_content):
        if index != len(pred_content)-1 and target in ['<mos>', '<eos>']:
            try:
                if (target_content[index-1]!=pred_content[index-1]):
                    print(index)
                    print(target_content[index-1:index+1], pred_content[index-1:index+1])
                assert(target_content[index-1]==pred_content[index-1])
                left_context = target_content[index-1]
                right_context = ' ' + target_content[index+1]
                if rtl:
                    left_context = left_context[::-1]
                if det(left_context, right_context):
                    if target == '<eos>':
                        if pred == '<eos>':
                            correct_eos += 1
                        else:
                            context = ' '.join(target_content[max(0, index-10):min(index, len(target_content)-1)]).replace('<eos>', '').replace('<mos>','') + ' <mos> ' + ' '.join(target_content[index:min(len(target_content)-1, index+10)]).replace('<eos>', '').replace('<mos>','')
                            type_two.append(context)
                            incorrect_eos += 1
                    else:
                        if pred == '<eos>':
                            incorrect_mos += 1
                            context = ' '.join(target_content[max(0, index-10):min(index, len(target_content)-1)]).replace('<eos>', '').replace('<mos>','') + ' <eos> ' + ' '.join(target_content[index:min(len(target_content)-1, index+10)]).replace('<eos>', '').replace('<mos>','')
                            type_one.append(context)
                        else:
                            correct_mos += 1
            except Exception as e:
                print(e)
                print(f"{' '.join(target_content[index-5:index+5])}\t{' '.join(pred_content[index-5:index+5])}")
                exit()
        index += 1
                  
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rubric_file_path', type=str)
    parser.add_argument('pred_file_path', type=str)
    parser.add_argument('--rtl', action='store_true')
    parser.add_argument('--determiner_type', default='en', choices=['en', 'multilingual', 'all'])
 
    args = parser.parse_args()

    if args.determiner_type == "en":
        determiner = PunctuationSpace()
    elif args.determiner_type == 'multilingual':
        determiner = MultilingualPunctuation()
    else:
        determiner = Split()

    score(args.rubric_file_path, args.pred_file_path, determiner, rtl=args.rtl)
