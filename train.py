import sentencepiece as spm
import argparse
import subprocess, time, os
import shutil

class Error(Exception):
    pass

def train_sentence_piece(configuration):
    spm.SentencePieceTrainer.Train(('--input=%s --model_prefix=%s --vocab_size=%s --character_coverage=%s --model_type=%s') % (configuration.file_path, configuration.model_prefix, configuration.vocab_size, configuration.character_coverage, configuration.model_type))

def encode_sentence_piece(file_path, model_path, output_path):
    
    sp = spm.SentencePieceProcessor()    

    if not sp.Load(model_path):
        raise Error("model was not successfully loaded")

    #sp.SetEncodeExtraOptions('eos')

    output = open(output_path, 'w')

    with open(file_path, 'r') as inputFile:
        for line in inputFile:
            tokens = sp.EncodeAsPieces(line) + ["\n"]
            for t in tokens:
                output.write(t + " ")

    output.close()

def prep_files(directory, train_path, test_path):
    src = open(train_path, 'r').read()
    lines = src.split('\n')
    train = open(os.path.join(directory, 'train.txt'), 'w')
    valid = open(os.path.join(directory, 'valid.txt'), 'w')
    train.write('\n'.join(lines[:(len(lines)//2)]))
    valid.write('\n'.join(lines[(len(lines)//2):]))
    shutil.copyfile(test_path, os.path.join(directory, 'test.txt'))

def train_language_model(configuration):
    print(configuration.tokenize)
    if configuration.tokenize:
        print(configuration.tokenize)
        print("Preprocessing and tokenizing the text file: ")
        encode_sentence_piece(configuration.file_path, configuration.model_path, configuration.output_path)
        encode_sentence_piece('/exp/rwicks/ersatz/data/europarl/europarl-v9-test.en', configuration.model_path, '/exp/rwicks/ersatz/data/europarl/europarl-v9-test.encoded')
    
    prep_files('/exp/rwicks/ersatz/data/europarl/europarl-v9/', '/exp/rwicks/ersatz/data/europarl/europarl-v9-train.encoded', '/exp/rwicks/ersatz/data/europarl/europarl-v9-test.encoded')

    params = "python -u %s/main.py" % configuration.awd_path

    for var in vars(configuration):
        if var not in ['train', 'train_data', 'execute', 'awd_path', 'tokenize', 'model_path', 'output_path', 'file_path']:
            params += " --%s %s" % (var, vars(configuration)[var]) if vars(configuration)[var] is not None else ""
    
    print(params) 
    start = time.time()
    print("Calling AWD-LSTM training")
    subprocess.call(params, shell=True)   
    end = time.time()
    print(start)
    print(end)
    print(end-start)

