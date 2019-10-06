import argparse
import config, train
import sys, traceback

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="A Multi-lingual Sentence Boundary Detection Toolkit", epilog='''
    
                                    ''')

    parser.add_argument('mode', help="The mode in which to run the program. Options are 'train', 'test', and 'segment' ")
    parser.add_argument('-c', '--config', help="The path to the configuration file")

    args = parser.parse_args()

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
    elif mode == 'test':
        print("Lol. Not today son")
        sys.exit(-1)
    elif mode == 'segment':
        print("Try again soon :) ")
        sys.exit(-1)
    else:
        print("Invalid parameters passed. Please specify 'train', 'test', or 'segment'.")
        sys.exit(-1)


