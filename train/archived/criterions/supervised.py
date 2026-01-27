import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, cross_entropy

from utils import import_attr
from .rl import compute_returns


# deprecated
class FreeRecallSumMSELoss(nn.Module):
    def __init__(self, var_weight=1.0) -> None:
        super().__init__()
        self.var_weight = var_weight
    
    def forward(self, output, gt):
        loss = torch.sum((torch.sum(output, dim=0) - torch.sum(gt, dim=0)) ** 2)
        gt_argmax = torch.argmax(gt, dim=2)
        for i in range(gt_argmax.shape[1]):
            loss -= torch.sum(torch.var(output[:, i, gt_argmax[:, i]], dim=0) / (torch.mean(output[:, i, gt_argmax[:, i]], dim=0) 
                + torch.finfo(torch.float32).eps)) * self.var_weight
        return loss

# deprecated
class FreeRecallSumMSEMultipleOutputLoss(nn.Module):
    def __init__(self, loss="FreeRecallSumMSELoss", var_weight=1.0, output_weight=[0.5, 0.5]) -> None:
        super().__init__()
        self.var_weight = var_weight
        self.output_weight = output_weight
        self.loss_class = import_attr("train.criterions.{}".format(loss))(var_weight=var_weight)
    
    def forward(self, output, gt):
        loss = 0.0
        assert len(output) == len(self.output_weight)
        for i in range(len(output)):
            loss += self.loss_class(output[i], gt) * self.output_weight[i]
        return loss

# deprecated
class FreeRecallSumMSETrainEncodeLoss(nn.Module):
    def __init__(self, loss="FreeRecallSumMSELoss", var_weight=1.0, output_weight=[0.5, 0.5], encode_weight=1.0, only_encode=False) -> None:
        super().__init__()
        self.var_weight = var_weight
        self.output_weight = output_weight
        self.encode_weight = encode_weight
        self.only_encode = only_encode
        self.loss_class = import_attr("train.criterions.{}".format(loss))(var_weight=var_weight)
    
    def forward(self, output, gt):
        loss = 0.0
        if len(output) == 2:
            free_recall_output_index = [0]
            encode_output_index = 1
        elif len(output) == 4:
            free_recall_output_index = [0, 1]
            encode_output_index = 2
        else:
            raise AttributeError("output length must be 2 or 4")
        if not self.only_encode:
            for i in free_recall_output_index:
                loss += self.loss_class(output[i], gt) * self.output_weight[i]
        loss += mse_loss(output[encode_output_index], gt) * self.encode_weight
        return loss

# deprecated
class FreeRecallSumCETrainEncodeLoss(nn.Module):
    def __init__(self, loss="FreeRecallSumMSELoss", var_weight=1.0, output_weight=[0.5, 0.5], encode_weight=1.0, only_encode=False) -> None:
        super().__init__()
        self.var_weight = var_weight
        self.output_weight = output_weight
        self.encode_weight = encode_weight
        self.only_encode = only_encode
        self.loss_class = import_attr("train.criterions.{}".format(loss))(var_weight=var_weight)
    
    def forward(self, output, gt):
        loss = 0.0
        if len(output) == 2:
            free_recall_output_index = [0]
            encode_output_index = 1
        elif len(output) == 4:
            free_recall_output_index = [0, 1]
            encode_output_index = 2
        else:
            raise AttributeError("output length must be 2 or 4")
        if not self.only_encode:
            for i in free_recall_output_index:
                loss += self.loss_class(output[i], gt) * self.output_weight[i]
        loss += cross_entropy(output[encode_output_index].reshape(-1, output[encode_output_index].shape[-1]), torch.argmax(gt, dim=2).reshape(-1)) * self.encode_weight
        return loss
    

class EncodingCrossEntropyLoss(nn.Module):
    def __init__(self, class_num, no_action_weight=1.0) -> None:
        super().__init__()
        self.class_num = class_num
        self.class_weights = torch.ones(class_num)
        self.class_weights[-1] = no_action_weight
        # self.class_weights[-2] = no_action_weight
    
    def forward(self, output, gt):
        # if self.phase == 'encoding':
        #     loss = cross_entropy(output[:memory_num].reshape(-1, output[:memory_num].shape[-1]), gt[:memory_num].reshape(-1), weight=self.class_weights)
        # elif self.phase == 'recall':
        #     loss = cross_entropy(output[memory_num:].reshape(-1, output[memory_num:].shape[-1]), gt[memory_num:].reshape(-1), weight=self.class_weights)
        # print(output.shape, gt.shape)
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

    def forward(self, output, gt):
        # assert self.nback < memory_num
        # if self.phase == 'encoding':
        #     loss = cross_entropy(output[self.nback:memory_num].reshape(-1, output[self.nback:memory_num].shape[-1]), 
        #                          gt[:memory_num-self.nback].reshape(-1), weight=self.class_weights)
        # elif self.phase == 'recall':
        #     loss = cross_entropy(output[memory_num:].reshape(-1, output[memory_num:].shape[-1]), 
        #                          gt[memory_num-self.nback:-self.nback].reshape(-1), weight=self.class_weights)
        loss = cross_entropy(output[self.nback:].reshape(-1, output[self.nback:].shape[-1]), gt[:-self.nback].reshape(-1), weight=self.class_weights)
        return loss

