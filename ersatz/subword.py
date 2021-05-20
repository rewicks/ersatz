# *-* coding: utf-8 *-*
import torch

class Vocabulary():

    def __init__(self):
        self.itos = ['<unk>', '<eos>', '<mos>', '<pad>']
        self.stoi = {'<unk>': 0, '<eos>': 1, '<mos>': 2, '<pad>': 3}

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, file_path):
        with open(file_path) as i:
            for line in i:
                word = line.split()[0].strip()
                self.add_word(word)

    def add_word(self, word):
        if word not in self.stoi:
            self.stoi[word] = len(self.itos)
            self.itos.append(word)

    def embed_word(self, word):
        return self.stoi.get(word, 2)

    def get_word(self, embedding):
        return self.itos[embedding]

    def detokenize(self, input_string):
        input_string = input_string.replace(' ', '')
        input_string = input_string.replace('\u2581', ' ')
        return input_string

    def encode(self, input_string, out_type=int):
        if out_type is int:
            arr = []
            input_string = input_string.split()
            for s in input_string:
                arr.append(self.embed_word(s))
            return arr
        else:
            return input_string.split()

    def decode(self, input_array):
        output = []
        for i in input_array:
            output.append(self.get_word(i))
        return ' '.join(output)

    def tensor_to_string(self, tensors):
        if len(tensors.shape) > 1:
            output = []
            for tens in tensors:
                output.append(self.decode(tens.tolist()))
            return output
        else:
            return self.decode(tensors.tolist())

    def context_to_tensor(self, contexts):
        con_arr = []
        fact_arr = []
        lab_arr = []
        for left, left_stream, right, right_stream, label in contexts:
            tens = []
            for l in left.split():
                tens.append(self.embed_word(l))
            for r in right.split():
                tens.append(self.embed_word(r))
            con_arr.append(tens)

            fact_arr.append(left_stream + right_stream)

            if label == "<eos>":
                lab_arr.append(0)
            else:
                lab_arr.append(1)
        return torch.tensor(con_arr), torch.tensor(fact_arr), torch.tensor(lab_arr)

class SentencePiece(Vocabulary):
    """
    Implements SentencePiece.
    https://github.com/google/sentencepiece/blob/master/python/README.md
    """
    def __init__(self, serialization=None, model_path=None, vocab_path = None, sample: bool = True, alpha: float = 0.5):
        import sentencepiece as spm
        if serialization is None:
            self.model = spm.SentencePieceProcessor()
            self.model_path = model_path
            if model_path is not None:
                self.model.Load(model_path)
            if vocab_path:
                self.vocab_path = vocab_path
                self.model.LoadVocabulary(vocab_path)
        else:
            self.model = spm.SentencePieceProcessor(model_proto=serialization)
        self.alpha = alpha
        self.sample = sample

    def __len__(self):
        return self.model.get_piece_size()

    def embed_word(self, word):
        return self.model[word]

    def encode(self, sentence, out_type=int) -> str:
        return self.model.encode(sentence, out_type=out_type)
        if out_type is int:
            if self.sample:
                return ' '.join(self.model.SampleEncodeAsPieces(sentence, nbest_size=1, alpha=self.alpha))
            else:
                return ' '.join(self.model.EncodeAsPieces(sentence))

    def decode(self, ids):
        return self.model.decode(ids)

    def merge(self, sentence: str, technique='replace') -> str:
        if technique == 'replace':
            return sentence.replace(' ', '').replace('▁', ' ')
        else:
            return self.model.decode(sentence)

def get_tokenizer(model_path, sample = False):
    return SentencePiece(model_path=model_path, sample=sample)
