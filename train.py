import sentencepiece as spm
import argparse
import subprocess, time, os
import shutil
import awd

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

def train_language_model(args):

    print("Calling AWD-LSTM training")
    awd.main(args)
    
