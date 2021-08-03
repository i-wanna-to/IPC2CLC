import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union, Tuple, List, Iterable, Dict
import logging
import gzip
from tqdm import tqdm
import numpy as np
import os
import json
from ..util import import_from_string, fullname, http_get
from .tokenizer import WordTokenizer, WhitespaceTokenizer


class BilinearCNN(nn.Module):
    """BilinearCNN-layer with multiple kernel-sizes over the word embeddings"""

    def __init__(self, in_word_embedding_dimension: int, n_filters: int = 128, filter_sizes: List[int] = [1, 3, 5], out_word_embedding_dimension = 768, dropout_rate = 0.1):
        nn.Module.__init__(self)
        self.config_keys = ['in_word_embedding_dimension', 'n_filters', 'filter_sizes', 'out_word_embedding_dimension', 'dropout_rate']
        self.in_word_embedding_dimension = in_word_embedding_dimension
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.out_word_embedding_dimension = out_word_embedding_dimension
        self.dropout_rate = dropout_rate

        self.embeddings_dimension = out_word_embedding_dimension

        # 下面的三个卷积　是一起作用在第一层的，对输入的第一批数据进行的，故此所有的卷积的in_channels＝１
        self.conv_0 = nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                    kernel_size = (filter_sizes[0], in_word_embedding_dimension))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                    kernel_size = (filter_sizes[1], in_word_embedding_dimension))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                    kernel_size = (filter_sizes[2], in_word_embedding_dimension))
        
        self.classifiers = nn.Linear((n_filters**2) * len(filter_sizes), out_word_embedding_dimension)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, features):
        token_embeddings = features['token_embeddings']
        token_embeddings = token_embeddings.unsqueeze(1) 
        # token_embeddings = [batch_size, 1, sen_len, emb_dim]
        
        conved_0 = F.relu(self.conv_0(token_embeddings).squeeze(3))
        conved_1 = F.relu(self.conv_1(token_embeddings).squeeze(3))
        conved_2 = F.relu(self.conv_2(token_embeddings).squeeze(3))
        # conved_n = [batch_size, n_filters, sen_len - filter_sizes[n] + 1]

        pooled_0 = self.bilinear_pooling(conved_0)
        pooled_1 = self.bilinear_pooling(conved_1)
        pooled_2 = self.bilinear_pooling(conved_2)
        # pooled_n = [batch_size, n_filters * n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
        # cat = [batch_size, n_filters * len(filter_sizes)]

        cat = self.classifiers(cat)

        features.update({'sentence_embedding': cat})
        
        return features

    def bilinear_pooling(self, conved_n):
        x = conved_n
        batch_size = x.size(0)
        feature_size = x.size(2)
        # x = x.view(batch_size , n_filters, feature_size)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x)*torch.sqrt(torch.abs(x)+1e-10))
        # x = [batch_size, n_filters * n_filters]

        return x

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError()

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'bilinearCnn_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'bilinearCnn_config.json'), 'r') as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model = BilinearCNN(**config)
        model.load_state_dict(weights)
        return model

