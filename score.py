import argparse, numpy

def stream(path):
    with open(path, 'r') as f:
        for line in f:
            for c in line:
                yield c

def score(correct_path, split_path):
    correct = stream(correct_path)
    split = stream(split_path)

    results = numpy.zeros((2,2))
    type_one = open('type_one.txt', 'a+')
    type_two = open('type_two.txt', 'a+')

    context = ['' for x in range(50)]

    EOF = False
    while (not EOF):
        try:
            c = next(correct)
            if c != ' ':
                s = next(split)
                context.pop(0)
                context.append(s)
                while s == ' ':
                    s = next(split)
                    context.pop(0)
                    context.append(s)
                if c == '\n' and s == '\n':
                    results[0][0] += 1
                elif c == '\n' and s != '\n':
                    c= next(correct)
                    while c != s:
                        c = next(correct)
                    results[0][1] += 1
                    type_two.write(''.join(context) + '\n\n')
                elif c != '\n' and s == '\n':
                    results[1][0] += 1
                    type_one.write(''.join(context) + '\n')
                    while s != c:
                        s = next(split)
                        context.pop(0)
                        context.append(s)
                else:
                    results[1][1] += 1
        except:
            EOF = True

    return results

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('map_file')

    args = parser.parse_args()

    results = numpy.zeros((2,2))

    with open(args.map_file, 'r') as f:
        for line in f:
            line = line.split()
            r = score(line[1], line[0])    
            print(f'Results on {line[1]}')
            print(r)
            results += r
    
    print("Overall Results:")
    print(results)
    acc = (results[0][0]+results[1][1])
    acc /= (results[0][0]+results[0][1]+results[1][0]+results[1][1])

    prec = (results[0][0]/(results[0][0] + results[1][0]))
    recall = (results[0][0]/(results[0][0]+results[0][1]))
    print(f'Accuracy (character level--heavily inflated): {acc}')
    print(f'Precision: {prec}')
    print(f'Recall: {recall}') 
