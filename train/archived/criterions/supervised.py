import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, cross_entropy

from utils import import_attr
from .rl import compute_returns
    

class EncodingCrossEntropyLoss(nn.Module):
    def __init__(self, class_num, no_action_weight=1.0) -> None:
        super().__init__()
        self.class_num = class_num
        self.class_weights = torch.ones(class_num)
        self.class_weights[-1] = no_action_weight
        # self.class_weights[-2] = no_action_weight

    def forward(self, output, gt, device="cpu"):
        # if self.phase == 'encoding':
        #     loss = cross_entropy(output[:memory_num].reshape(-1, output[:memory_num].shape[-1]), gt[:memory_num].reshape(-1), weight=self.class_weights)
        # elif self.phase == 'recall':
        #     loss = cross_entropy(output[memory_num:].reshape(-1, output[memory_num:].shape[-1]), gt[memory_num:].reshape(-1), weight=self.class_weights)
        # print(output.shape, gt.shape)
        self.class_weights = self.class_weights.to(device)
        loss = cross_entropy(output.reshape(-1, output.shape[-1]), gt.reshape(-1), weight=self.class_weights)
        return loss
    

class EncodingNBackCrossEntropyLoss(nn.Module):
    def __init__(self, class_num, no_action_weight=1.0, nback=1) -> None:
        super().__init__()
        self.class_num = class_num
        self.nback = nback
        self.class_weights = torch.ones(class_num)
        self.class_weights[-1] = no_action_weight
        self.class_weights[-2] = no_action_weight

    def forward(self, output, gt, device="cpu"):
        self.class_weights = self.class_weights.to(device)
        # assert self.nback < memory_num
        # if self.phase == 'encoding':
        #     loss = cross_entropy(output[self.nback:memory_num].reshape(-1, output[self.nback:memory_num].shape[-1]), 
        #                          gt[:memory_num-self.nback].reshape(-1), weight=self.class_weights)
        # elif self.phase == 'recall':
        #     loss = cross_entropy(output[memory_num:].reshape(-1, output[memory_num:].shape[-1]), 
        #                          gt[memory_num-self.nback:-self.nback].reshape(-1), weight=self.class_weights)
        loss = cross_entropy(output[self.nback:].reshape(-1, output[self.nback:].shape[-1]), gt[:-self.nback].reshape(-1), weight=self.class_weights)
        return loss

