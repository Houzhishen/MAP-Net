import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class EmbeddingWord(nn.Module):
    def __init__(self,
                 indim,
                 emb_size,
                 drop_rate):
        super(EmbeddingWord, self).__init__()

        self.emb_size = emb_size

        self.layer1 = nn.Sequential(nn.Linear(indim, 300),
                                    # nn.BatchNorm1d(512),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(300, self.emb_size))
        self.dropout = nn.Dropout(drop_rate)
        

    def forward(self, x):
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)

        return out

