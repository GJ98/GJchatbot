from typing import Dict

import torch
from torch import nn

from GJchatbot.model.layers.conv import Conv
from GJchatbot.model.layers.linear import Linear


class CNN(nn.Module):

    def __init__(self, label_dict: Dict[int, str]):
        """의도 분류 CNN 모델 클래스

        Args:
            label_dict (Dict[int, str]): 의도 라벨 딕셔너리
        """
        
        super().__init__()

        self.label_dict = label_dict

        self.vector_size = 128
        self.hidden_size = 512
        self.maxlen = 8
        self.label_size = len(label_dict)

        '''
        self.conv, self.maxpool = [], []
        for idx in range(3):
            self.conv.append(Conv(in_dim=self.vector_size,
                                  out_dim=self.hidden_size,
                                  kernel_size=3 + idx))
            self.maxpool.append(nn.MaxPool1d(kernel_size=self.maxlen-(2 + idx)))
        '''
        for idx in range(3):
            setattr(self, 'conv{}'.format(idx), Conv(in_dim=self.vector_size,
                                                     out_dim=self.hidden_size,
                                                     kernel_size=3+idx))
            setattr(self, 'maxpool{}'.format(idx), nn.MaxPool1d(kernel_size=self.maxlen-(2+idx)))
        
        self.linear1 = Linear(in_dim=3*self.hidden_size, 
                              out_dim=self.hidden_size, 
                              acti='relu')
        self.linear2 = Linear(in_dim=self.hidden_size, 
                              out_dim=self.label_size, 
                              acti='softmax')

        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.long()
        embed_x = x.permute(0, 2, 1)

        '''
        concat_x = [] 
        for idx in range(3):
            conv_x = self.conv[idx](embed_x)
            pool1_x = self.maxpool[idx](conv_x)
            concat_x.append(pool1_x)
        '''

        conv0_x = self.conv0(embed_x)
        conv1_x = self.conv1(embed_x)
        conv2_x = self.conv2(embed_x)

        pool0_x = self.maxpool0(conv0_x)
        pool1_x = self.maxpool1(conv1_x)
        pool2_x = self.maxpool2(conv2_x)

        concat_x = torch.cat([pool0_x, pool1_x, pool2_x], dim=1)
        concat_x = concat_x.view(concat_x.size(0), concat_x.size(1))

        linear1_x = self.linear1(concat_x)
        linear1_x = self.dropout(linear1_x)

        linear2_x = self.linear2(linear1_x)
        return linear2_x
