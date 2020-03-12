import argparse, numpy

def stream(path):
    content = open(path).read()
    while content != content.replace('  ', ' '):
        content = content.replace('  ', ' ')
    content = content.replace(' ', ' <mos> ')
    content = content.replace('\n', ' <eos> ')
    f = content.split()    
    for index, word in enumerate(f):
        if word == '<eos>' or word == '<mos>':
            yield word, f[index-10:index+10]

def score(split_path):
    language = split_path.split('/')[1]
    correct_path = '/home/hltcoe/rwicks/ersatz/data/raw/' + language + '/test/' + '/'.join(split_path.split('/')[2:])   

    correct = stream(correct_path)
    split = stream(split_path)

    results = numpy.zeros((2,2))
    type_one = open('type_one.txt', 'a+')
    type_two = open('type_two.txt', 'a+')

    EOF = False
    while (not EOF):
        try:
            c, debug_context = next(correct)
            
            s, context = next(split)
        
            if c == '<eos>' and s == '<eos>':
                results[0][0] += 1
            elif c == '<eos>' and s != '<eos>':
                results[0][1] += 1
                type_two.write('Corr: ' + ' '.join(debug_context) + '\nPred: ' + ' '.join(context) + '\n\n')
            elif c != '<eos>' and s == '<eos>':
                results[1][0] += 1
                type_one.write('Corr: ' + ' '.join(debug_context) + '\nPred: ' + ' '.join(context) + '\n\n')
            else:
                results[1][1] += 1
        except:
            EOF = True
    return results

'''
def score(correct_path, split_path):
    correct = stream(correct_path)
    split = stream(split_path)

    results = numpy.zeros((2,2))
    type_one = open('type_one.txt', 'a+')
    type_two = open('type_two.txt', 'a+')

    type_one.write(split_path + '\n')
    type_two.write(split_path + '\n')
    context = ['' for x in range(50)]
    debug_context = ['' for x in range(50)]
    EOF = False
    while (not EOF):
        try:
            c = next(correct)
            debug_context.pop(0)
            debug_context.append(c)
            while c != ' ' and c != '\n':
                c = next(correct)
                debug_context.pop(0)
                debug_context.append(c)
            
            s = next(split)            
            context.pop(0)
            context.append(s)
            while s != ' ' and s != '\n':
                s = next(split)
                context.pop(0)
                context.append(s)            

            if c == '\n' and s == '\n':
                results[0][0] += 1
            elif c == '\n' and s != '\n':
                results[0][1] += 1
                type_two.write(''.join(context) + '\n\n')
            elif c != '\n' and s == '\n':
                results[1][0] += 1
                type_one.write(''.join(context) + '\n')
            else:
                results[1][1] += 1
        except:
            EOF = True
    return results
'''
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('map_file')
    parser.add_argument('split_files', nargs='*')

    args = parser.parse_args()

    results = numpy.zeros((2,2))

    scores = open('scores.txt', 'a+')

    for fi in args.split_files:
        r = score(fi)
        print(fi)
        print(r)
        prec = (r[0][0]/(r[0][0]+r[1][0]))*100
        recall = (r[0][0]/(r[0][0]+r[0][1]))*100
        f1 = 2*(prec*recall)/(prec+recall)
        print(f'{fi}\t{prec:.1f},{recall:.1f},{f1:.1f}')
        #print(f'Recall : {recall}')
        #print(f'Precision : {prec}')
        #print(f'Accuracy : {((r[0][0]+r[1][1])/(r[0][0] + r[1][0] + r[1][1]))*100}')
        results += r
        #(f'{fi}\t{prec:.1f}\t{recall:.1f}\t{f1:.1f}') 
        scores.write(f'{fi}\t{prec:.1f}\t{recall:.1f}\t{f1:.1f}\n')
    print("Overall Results:")
    print(results)
    acc = (results[0][0]+results[1][1])
    acc /= (results[0][0]+results[0][1]+results[1][0]+results[1][1])

    prec = (results[0][0]/(results[0][0] + results[1][0]))
    recall = (results[0][0]/(results[0][0]+results[0][1]))
    print(f'Accuracy (character level--heavily inflated): {acc}')
    print(f'Precision: {prec}')
    print(f'Recall: {recall}') 
