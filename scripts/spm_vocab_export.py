import json
import sys

vocab = {}


vocab['<eos>'] = 0
vocab['<mos>'] = 1
vocab['<unk>'] = 2
vocab['<hole>'] = 3
vocab['<pad>'] = 4

with open(sys.argv[1]) as inputFile:
    for line in inputFile:
        word = line.split()[0].strip()
        if word not in vocab:
            vocab[word] = len(vocab)

with open(sys.argv[2], 'w') as outputFile:
    json.dump(vocab, outputFile, indent=4)
