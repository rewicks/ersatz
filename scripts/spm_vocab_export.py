import json
import sys

vocab = {}
with open(sys.argv[1]) as inputFile:
    for line in inputFile:
        word = line.split()[0].strip()
        vocab[word] = len(vocab)

for key in vocab:
    vocab[key] += 4

vocab['<eos>'] = 0
vocab['<mos>'] = 1
vocab['<unk>'] = 2
vocab['<hole>'] = 3
vocab['<pad>'] = 4

with open(sys.argv[1], 'w') as outputFile:
    json.dump(vocab, outputFile, indent=4)
