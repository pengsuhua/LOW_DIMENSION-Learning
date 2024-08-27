from torch import nn

## 原始的MLP结构
# class MLPModel(nn.Module):
#
#     def __init__(self):
#         super(MLPModel, self).__init__()
#         self.linear1 = nn.Linear(2, 1024)
#         #模型定义中加入Dropout：防止过拟合：
#         # self.dropout = nn.Dropout(p=0.2)  # 0.5 是 Dropout 概率
#
#         self.linear2 = nn.Linear(1024, 128)
#         # self.linear3 = nn.Linear(128, 128)
#         # self.linear4 = nn.Linear(256, 256)
#         # self.linear5 = nn.Linear(256, 256)
#         # self.linear6 = nn.Linear(256, 256)
#
#
#
#         self.linear_out = nn.Linear(128, 1)
#         # self.act = nn.Sigmoid()
#         self.act = nn.ReLU()
#         self.num = self.num_parameters()
#         print(f'number of model parameters is {self.num}')
#
#     def forward(self, x):
#         # out = self.flatten(x)
#         out = self.linear1(x)
#         out = self.act(out)
#
#         out = self.linear2(out)
#         out = self.act(out)
#
#         # out = self.linear3(out)
#         # out = self.act(out)
#         #
#         # out = self.linear4(out)
#         # out = self.act(out)
#         #
#         # out = self.linear5(out)
#         # out = self.act(out)
#
#
#         # out = self.linear6(out)
#         # out = self.act(out)
#
#
#
#
#         out = self.linear_out(out)
#         out = self.act(out)
#         return out
#
#     def num_parameters(self):
#         ps = self.parameters()
#         num = sum(p.numel() for p in ps if p.requires_grad)
#         return num


### 更加复杂的MLP结构
'''

网络层数增加：我将模型深度增加到5层线性层，但在每一层减少了神经元数量，这样可以保持总参数数量在合理范围内。

Batch Normalization：在每个线性层后面添加了 Batch Normalization，它可以加快训练收敛速度并改善模型性能，而几乎不增加参数数量。

Dropout：在最后一层全连接层之前引入了 Dropout，以减少过拟合。

ReLU 激活函数：每个线性层后都应用了 ReLU 激活函数，增加模型的非线性表达能力。

'''

import torch
import torch.nn as nn


class MLPModel(nn.Module):

    def __init__(self):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.linear5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.linear_out = nn.Linear(32, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.num = self.num_parameters()
        print(f'number of model parameters is {self.num}')

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.linear3(out)
        out = self.bn3(out)
        out = self.act(out)

        out = self.linear4(out)
        out = self.bn4(out)
        out = self.act(out)

        out = self.linear5(out)
        out = self.bn5(out)
        out = self.act(out)

        # out = self.dropout(out)  # Dropout 加在最后一层全连接层之前

        out = self.linear_out(out)
        return out

    def num_parameters(self):
        ps = self.parameters()
        num = sum(p.numel() for p in ps if p.requires_grad)
        return num


