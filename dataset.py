import argparse
import os

def split_train_file(file_path, eos=False, context_size=4):
    left_contexts = []
    right_contexts = []
    labels = []

    content = open(file_path).read()
    content = content.replace('\n', ' <eos> ')
    content = content.split()

    for index, word in enumerate(content):
        if (content[index-1] != '<eos>') or eos:
            left_temp = []
            right_temp = []
            # Get the left context
            temp_index = index - 1
            while (len(left_temp) < context_size):
                if temp_index >= 0:
                    if not eos:
                        if content[temp_index] != '<eos>':
                            left_temp.append(content[temp_index])
                    else:
                        left_temp.append(content[temp_index])
                else:
                    left_temp.append('<PAD>')
                temp_index -= 1
        
            left_temp.reverse()
            left_contexts.append(left_temp)

            # Get the label
            labels.append(word)

            # Get the right context
            temp_index = index + 1
            while (len(right_temp) < context_size):
                if temp_index < len(content):
                    if content[temp_index] != '<eos>':
                        right_temp.append(content[temp_index])
                else:
                    right_temp.append('<PAD>') 
                temp_index += 1

            right_contexts.append(right_temp)

    return left_contexts, right_contexts, labels

def write_training_files(file_path, left_contexts, right_contexts, labels, context_size):
    output_path = file_path.split('/')[-1].split('.')
    output_path = output_path[:-1] + [str(context_size)+'-context'] + output_path[-1:]
    output_path = os.path.join('/'.join(file_path.split('/')[:-1]),'.'.join(output_path))

    with open(output_path, 'w') as f:
        for left, right, label in zip(left_contexts, right_contexts, labels):
            f.write(' '.join(left) + ' ||| ' + ' '.join(right) + ' ||| ' + label + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--context-size', type=int)
    
    args = parser.parse_args()
    
    left, right, labels = split_train_file(args.path, context_size=args.context_size)
    write_training_files(args.path, left, right, labels, args.context_size)
