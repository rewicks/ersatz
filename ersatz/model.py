import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ErsatzTransformer(nn.Module):

    def __init__(self, tokenizer, args):
        super(ErsatzTransformer, self).__init__()

        self.factor_embed_size = 0
        self.source_factors = False
        if 'source_factors' in args and args.source_factors:
            self.fact_emb = nn.Embedding(6, args.factor_embed_size)
            self.factor_embed_size = args.factor_embed_size
            self.source_factors = True

        if args.transformer_nlayers > 0:
            self.transformer = True
            # each layer of the transformer
            encoder_layer = nn.TransformerEncoderLayer(args.embed_size+self.factor_embed_size, args.nhead, dropout=args.dropout)
            # build the transformer with n of the previous layers
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.transformer_nlayers)
            self.pos_embed = PositionalEncoding(embed_size=args.embed_size+self.factor_embed_size, dropout=args.dropout, max_len=(args.left_size + args.right_size))
            self.nhead = args.nhead

        else:
            self.transformer = False
        # embeds the input into a embed_size dimensional space
        self.src_emb = nn.Embedding(len(tokenizer), args.embed_size)
        self.embed_dropout = nn.Dropout(args.dropout)

        # vocab; includes stoi and itos look ups
        self.tokenizer = tokenizer
        self.transformer_nlayers = args.transformer_nlayers
        self.linear_nlayers = args.linear_nlayers
        self.dropout = args.dropout
        self.left_context_size = args.left_size
        self.right_context_size = args.right_size
        self.embed_size = args.embed_size
        self.max_size = args.left_size + args.right_size
        self.args = args
        self.generator = Generator(args.embed_size+self.factor_embed_size, self.max_size,
                                   nlayers=args.linear_nlayers, activation_type=args.activation_type)

    def forward(self, src, factors=None):
        if self.transformer:
            src = src.t()
            src = self.src_emb(src)
            if factors is not None:
                factors = factors.t()
                factors = self.fact_emb(factors)
                src = torch.cat((src, factors), dim=2)
            embed = self.pos_embed(src * math.sqrt(self.embed_size+self.factor_embed_size))
            embed = self.encoder(embed).transpose(0,1)
        else:
            embed = self.src_emb(src)
            if factors is not None:
                factors = self.fact_emb(factors)
                embed = torch.cat((embed, factors), dim=2)
            #embed = self.embed_dropout(embed)
        #output = self.embed_dropout(embed)
        return self.generator(embed)

class Generator(nn.Module):
    
    # could change this to mean-pool or max pool
    def __init__(self, embed_size, max_size, nlayers=0, activation_type="tanh"):
        super(Generator, self).__init__()
        hidden = max_size * embed_size

        if activation_type == 'tanh':
            activation = nn.Tanh()
        if nlayers > 0:
            hidden_layers = [
                nn.Linear(hidden, embed_size),
                activation
            ]
            for n in range(1, nlayers):
                hidden_layers.append(
                    nn.Linear(embed_size, embed_size)
                )
                hidden_layers.append(
                    activation
                )
            self.hidden_layers = nn.ModuleList(hidden_layers)
            self.proj = nn.Linear(embed_size, 2)
        else:
            self.hidden_layers = None
            self.proj = nn.Linear(hidden, 2)

    def forward(self, x):
        x = x.reshape(x.size()[0], -1)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                x = layer(x)
        x = self.proj(x)
        return F.log_softmax(x, dim=-1)

class PositionalEncoding(nn.Module):
    
    def __init__(self, embed_size=512, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

