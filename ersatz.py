import argparse
import config, train
import sys, traceback
import glob
import test

def train_model(args):
    if args.model == 'spm':
        if args.config is not None:
            try:
                print("Reading in configuration file at %s" % args.config)
                configuration = config.TokenConfiguration((open(args.config, 'r')).read())
                print("Training SentencePiece with the following configuration: \n" + str(configuration))
                train.train_sentence_piece(configuration)
                output_path = configuration.file_path.split('/')[-1].split('.')
                output_path = 'proc/' + ''.join(output_path[0:-1]) + '.tok.' + output_path[-1]
                train.encode_sentence_piece(configuration.file_path, configuration.model_prefix + '.model', output_path)
            except Exception as e:
                print(e)
                print("Something went wrong while trying to train a Sentencepiece Model with the following configuration file:")
                print(configuration)
    elif args.model == 'awd':
        if args.config is not None:
            try:
                print(f'Reading in configuration file at {args.config}')
                configuration = config.ModelConfiguration((open(args.config, 'r')).read())
                print(f'Training AWD-LSTM Model with the following configuration: \n {configuration}')
                train.train_language_model(configuration)
            except Exception as e:
                print(e.traceback())
                print("Something went wrong while trying to train an AWD-LSTM Model with the following configuration file:")
                print(configuration)


def test_model(args):
    print("Lol. Not today son")
    sys.exit(-1)

def segment(args):
    if args.config is not None:
        configuration = config.SegmentConfiguration((open(args.config, 'r')).read())
        segmenter = test.EvalModel(configuration.language_model_path, configuration.tokenizer_path)
        for f in glob.glob(configuration.test_files):
            segmenter.evaluate(f)    

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="A Multi-lingual Sentence Boundary Detection Toolkit", epilog='''
    
                                    ''')

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train', help='training mode')
    train_parser.add_argument('model', help="Which model to train. Options are awd or spm")
    train_parser.add_argument('-c', '--config', help="The path to the configuration file")
    train_parser.set_defaults(func=train_model)


    test_parser = subparsers.add_parser('test', help='testing mode')
    test_parser.add_argument('-c', '--config', help="The path to the configuration file")
    test_parser.set_defaults(func=test_model)

    seg_parser = subparsers.add_parser('segment', help='segment mode')
    seg_parser.add_argument('-c', '--config', help="The path to the configuration file")
    seg_parser.set_defaults(func=segment)

    args = parser.parse_args()

    args.func(args)


'''
    if args.mode == 'train':
        if args.config is not None:
            try:
                print("Reading in configuration file at %s" % args.config)
                configuration = config.Configuration((open(args.config, 'r')).read())
                if configuration.token_configuration is not None:
                    if configuration.token_configuration.execute:
                        print("Training SentencePiece with the following configuration: \n" + str(configuration.token_configuration))
                        train.train_sentence_piece(configuration.token_configuration)
                if configuration.model_configuration is not None:
                    if configuration.model_configuration.execute:
                        print("Training AWD-LSTM with the following configuration: \n" + str(configuration.model_configuration))
                        train.train_language_model(configuration.model_configuration)
            except Exception as e:
                print("Could not open configuration file specified at path %s. Please try again." % (args.config))
                print(e)
                traceback.print_exc(file=sys.stdout)
                sys.exit(-1)
        else:
            print("Running without a configuration file is currently not supported. Please see an example at default.config and try again :)")
            sys.exit(-1)
'''

