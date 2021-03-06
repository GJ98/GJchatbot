from typing import Dict

import torch
from torch import nn

from GJchatbot.model.layers.linear import Linear


class LSTM(nn.Module):
    def __init__(self, label_dict: Dict[int, str]):
        """개채명 분류 LSTM 모델 클래스

        Args:
            label_dict (Dict[int, str]): 개채명 라벨 딕셔너리
        """

        super().__init__()

        self.label_dict = label_dict

        self.vocab_size = 128
        self.hidden_size = 512
        self.num_layers = 1
        self.label_size = len(label_dict)

        self.lstm = nn.LSTM(input_size=self.vocab_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.linear = nn.Linear(in_features=2*self.hidden_size, 
                                out_features=self.label_size)

        self.dropout = nn.Dropout()

    def forward(self, x):
        lstm, _ = self.lstm(x.float())

        linear = self.linear(lstm)

        return linear
