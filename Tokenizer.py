import sentencepiece as spm
import torch
from error import Error

class Tokenizer():
    def __init__(self):
        self.model = None

    def load(self, model_path):
        return torch.load(model_path)


class SPM(Tokenizer):
    def __init__(self, model_path):
        super().__init__()
        self.model = spm.SentencePieceProcessor()
        if not self.model.Load(model_path):
            Error("model was not successfully loaded")

    def encode(self, string):
        return self.model.EncodeAsPieces(string)
