import json
#from prettytable import PrettyTable

class TokenConfiguration():
    def __init__(self, conf):
        conf = json.loads(conf) 
        # All the parameters for the SentencePiece tokenization training
        self.file_path = conf["file_path"] if "file_path" in conf else None
        self.model_prefix = conf["model_path"] if "model_path" in conf else None
        self.vocab_size = conf["vocab_size"] if "vocab_size" in conf else "8000"
        self.character_coverage = conf["character_coverage"] if "character_coverage" in conf else "1.0"
        self.model_type = conf["model_type"] if "model_type" in conf else "bpe"
    
    def __str__(self):
        
        printout = "\n-----------------------------------------------------------------------------------------------------\n"
        printout += "\tPath of training data file:      %s\n" % str(self.file_path)
        printout += "\tPrefix of the model file:        %s\n" % str(self.model_prefix)
        printout += "\tVocabulary Size:                 %s\n" % str(self.vocab_size)
        printout += "\tCharacter Coverage:              %s\n" % str(self.character_coverage)
        printout += "\tModel Type:                      %s\n" % str(self.model_type) 
        return printout
 
class ModelConfiguration():
    def __init__(self, conf):
        conf = json.loads(conf)
        
        # Local location of the awd-lstm project       
        self.train_path = conf["train_path"] if "train_path" in conf else None
        self.tokenizer_model_path = conf["tokenizer_model_path"] if "tokenizer_model_path" in conf else None

        # All the parameters to train the AWD-LSTM language model
        self.epochs = int(conf["epochs"]) if "epochs" in conf else None
        self.nlayers = int(conf["nlayers"]) if "nlayers" in conf else None
        self.emsize = int(conf["emsize"]) if "emsize" in conf else None
        self.nhid = int(conf["nhid"]) if "nhid" in conf else None
        self.alpha = float(conf["alpha"]) if "alpha" in conf else None
        self.beta = float(conf["beta"]) if "beta" in conf else None
        self.dropoute = float(conf["dropoute"]) if "dropoute" in conf else None
        self.dropouth = float(conf["dropouth"]) if "dropouth" in conf else None
        self.dropouti = float(conf["dropouti"]) if "dropouti" in conf else None
        self.dropout = float(conf["dropout"]) if "dropout" in conf else None
        self.wdrop = float(conf["wdrop"]) if "wdrop" in conf else None
        self.wdecay = float(conf["wdecay"]) if "wdecay" in conf else None
        self.bptt = int(conf["bptt"]) if "bptt" in conf else None
        self.batch_size = int(conf["batch_size"]) if "batch_size" in conf else None
        self.optimizer = conf["optimizer"] if "optimizer" in conf else None
        self.lr = float(conf["lr"]) if "lr" in conf else None
        self.save = conf["save"] if "save" in conf else None    
        self.when = int(conf["when"]) if "when" in conf else None
        self.model = conf["model"] if "model" in conf else None
        self.clip = float(conf["clip"]) if "clip" in conf else 0.25
        self.seed = int(conf["seed"]) if "seed" in conf else 1111
        self.nonmono = int(conf["nonmono"]) if "nonmono" in conf else 5
        self.log_interval = int(conf["log-interval"]) if "log-interval" in conf else 200
               
 
        if "resume" in conf and conf["resume"] == "True":
            self.resume = True
        else:
            self.resume = False      
 
        if "cuda" in conf and conf["cuda"] == "True":
            self.cuda = True
        else:
            self.cuda = False

        if "tied" in conf and conf["tied"] == "True":
            self.tied = True
        else:
            self.tied = False

    def __str__(self):
        printout = "\n--------------------------------------------------------------------------------------------------------\n"
        printout += "\tPath of training file:                    | %s\n" % str(self.train_path)
        printout += "\tPath of model file:                          | %s\n" % str(self.tokenizer_model_path)
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
