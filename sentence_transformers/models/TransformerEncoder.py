import torch
from torch import nn
from torch.autograd import Variable
from typing import List
import os
import json
import math



class PositionalEncoding(nn.Module):
    "Implement the PE function."
    # d_model == embedding_size
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        #0 -> 0.0 Modified 20190713
        position = torch.arange(0.0, max_len).unsqueeze(1) # [5000] -> [5000, 1]
        #0 -> 0.0 Modified 20190713
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # dim = 2 -> dimension = 3
        self.register_buffer('pe', pe)  # 计算好的位置信息，存储在pe中
        
    def forward(self, x):
        # x 应该为 word_embedding
        # 返回的 x = word_embedding (x) + positional_embedding
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder running over word embeddings.
    """
    def __init__(self, word_embedding_dimension: int, nhead: int = 8, num_layers: int = 6, dropout: float = 0):
        nn.Module.__init__(self)
        self.config_keys = ['word_embedding_dimension', 'nhead', 'num_layers', 'dropout']
        self.word_embedding_dimension = word_embedding_dimension
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        # PositionalEncoding
        self.pe = PositionalEncoding(self.word_embedding_dimension, self.dropout)
        # TransformerEncoder
        self.transEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.word_embedding_dimension, nhead=self.nhead), num_layers=self.num_layers)

    def forward(self, features):

        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        token_embeddings = self.pe(token_embeddings).transpose(0, 1)
        src_key_padding_mask = (attention_mask == 0)
        token_embeddings = self.transEncoder(token_embeddings, src_key_padding_mask = src_key_padding_mask).transpose(0, 1)

        features.update({'token_embeddings': token_embeddings})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.word_embedding_dimension

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError()

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'transformerEncoder_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'transformerEncoder_config.json'), 'r') as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        model = TransformerEncoder(**config)
        model.load_state_dict(weights)
        return model