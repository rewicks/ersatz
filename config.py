import json
#from prettytable import PrettyTable

class TokenConfiguration():
    def __init__(self, conf):
        # Whether or not the tokenization will be trained on this run
        if conf["execute"] == "True":
            self.execute = True
        else:
            self.execute = False
        
        conf = conf["configuration"]
        
        # All the parameters for the SentencePiece tokenization training
        self.file_path = conf["file_path"] if "file_path" in conf else None
        self.model_prefix = conf["model_path"] if "model_path" in conf else None
        self.vocab_size = conf["vocab_size"] if "vocab_size" in conf else "8000"
        self.character_coverage = conf["character_coverage"] if "character_coverage" in conf else "1.0"
        self.model_type = conf["model_type"] if "model_type" in conf else "bpe"
    
    def __str__(self):
        
        printout = "\n-------------------------------------------------\n"
        printout += "\tPath of training data file:      %s\n" % str(self.file_path)
        printout += "\tPrefix of the model file:        %s\n" % str(self.model_prefix)
        printout += "\tVocabulary Size:                 %s\n" % str(self.vocab_size)
        printout += "\tCharacter Coverage:              %s\n" % str(self.character_coverage)
        printout += "\tModel Type:                      %s\n" % str(self.model_type) 
        return printout
 
class ModelConfiguration():
    def __init__(self, conf):
        # Whether or not the language model will be trained on this run
        if conf["execute"] == 'True':
            self.execute = True
        else:
            self.execute = False
        
        conf = conf["configuration"]

        # Local location of the awd-lstm project       
        self.awd_path = conf["awd_path"] if "awd_path" in conf else "~/rwicks/awd-lstm-lm"
 
        # All the parameters to encode the file for training the language model
        # There must already exist a valid trained SentencePiece model
        if "tokenize" in conf:
            if conf["tokenize"] == 'True':
                self.tokenize = True
            else:
                self.tokenize = False
        else:
            self.tokenize = False
        self.file_path = conf["file_path"] if "file_path" in conf else None
        self.model_path = conf["model_path"] if "model_path" in conf else None
        self.output_path = conf["output_path"] if "output_path" in conf else None
        self.train = conf["train"] if "train" in conf else None
        self.test = conf["test"] if "test" in conf else None        

        # All the parameters to train the AWD-LSTM language model
        self.epochs = conf["epochs"] if "epochs" in conf else None
        self.nlayers = conf["nlayers"] if "nlayers" in conf else None
        self.emsize = conf["emsize"] if "emsize" in conf else None
        self.nhid = conf["nhid"] if "nhid" in conf else None
        self.alpha = conf["alpha"] if "alpha" in conf else None
        self.beta = conf["beta"] if "beta" in conf else None
        self.dropoute = conf["dropoute"] if "dropoute" in conf else None
        self.dropouth = conf["dropouth"] if "dropouth" in conf else None
        self.dropouti = conf["dropouti"] if "dropouti" in conf else None
        self.dropout = conf["dropout"] if "dropout" in conf else None
        self.wdrop = conf["wdrop"] if "wdrop" in conf else None
        self.wdecay = conf["wdecay"] if "wdecay" in conf else None
        self.bptt = conf["bptt"] if "bptt" in conf else None
        self.batch_size = conf["batch_size"] if "batch_size" in conf else None
        self.optimizer = conf["optimizer"] if "optimizer" in conf else None
        self.lr = conf["lr"] if "lr" in conf else None
        self.data = conf["train"] if "data" in conf else self.output_path
        self.save = conf["save"] if "save" in conf else None    
        self.when = conf["when"] if "when" in conf else None
        self.model = conf["model"] if "model" in conf else None

    def __str__(self):
        printout = "\n--------------------------------------------------------------------------------------------------------\n"
        printout += "\tTokenization:                                | %s\n" % str(self.tokenize)
        printout += "\tPath of file to tokenize:                    | %s\n" % str(self.file_path)
        printout += "\tPath of model file:                          | %s\n" % str(self.model_path)
        printout += "\tPrefix of location to save tokenized text:   | %s\n" % str(self.output_path)
        printout += "\tEpochs:                                      | %s\n" % str(self.epochs)
        printout += "\tNLayers:                                     | %s\n" % str(self.nlayers)
        printout += "\tEmsize:                                      | %s\n" % str(self.emsize)
        printout += "\tNHID:                                        | %s\n" % str(self.nhid)
        printout += "\tAlpha:                                       | %s\n" % str(self.alpha)
        printout += "\tBeta:                                        | %s\n" % str(self.beta)
        printout += "\tDropoutE:                                    | %s\n" % str(self.dropoute)
        printout += "\tDropoutH:                                    | %s\n" % str(self.dropouth)
        printout += "\tDropoutI:                                    | %s\n" % str(self.dropouti)
        printout += "\tDropout:                                     | %s\n" % str(self.dropout)
        printout += "\tWDrop:                                       | %s\n" % str(self.wdrop)
        printout += "\tWDecay:                                      | %s\n" % str(self.wdecay)
        printout += "\tBPTT:                                        | %s\n" % str(self.bptt)
        printout += "\tBatch Size:                                  | %s\n" % str(self.batch_size)
        printout += "\tOptimizer:                                   | %s\n" % str(self.optimizer)
        printout += "\tLR:                                          | %s\n" % str(self.lr)
        printout += "\tLocation of pre-processed text:              | %s\n" % str(self.data)
        printout += "\tPath of location to save model:              | %s\n" % str(self.save)
        printout += "\tWhen:                                        | %s\n" % str(self.when)
        printout += "\tModel Type:                                  | %s\n" % str(self.model)
        printout += "--------------------------------------------------------------------------------------------------------\n"
        return printout 

class Configuration():
    def __init__(self, configuration):
        conf = json.loads(configuration)
        self.token_configuration = TokenConfiguration(conf["train_token"]) if "train_token" in conf else None
        self.model_configuration = ModelConfiguration(conf["train_lm"]) if "train_lm" in conf else None
