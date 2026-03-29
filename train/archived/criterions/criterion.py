from math import gamma
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss, mse_loss
from torch.distributions import Categorical


class MultiSupervisedLoss(nn.Module):
    def __init__(self, criteria, output_index=[[0]]):
        """
        a list of criterions for mutliple outputs, all for supervised learning
        
        criteria: list of criterion
        output_index: list of int, the index of the output that the criterion is applied to
        """
        super().__init__()
        assert len(criteria) == len(output_index)
        self.criteria = criteria
        self.output_index = output_index
    
    def forward(self, outputs, gts, device="cpu"):
        """
        outputs: list of torch.tensor with a number of action_num, each` size is (timesteps * batch_size, output_dim)
        gts: list of torch.tensor, overall size is (timesteps * batch_size, action_num)
        """
        loss = None
        for i in range(len(self.criteria)):
            for j in range(len(self.output_index[i])):
                if loss is None:
                    loss = self.criteria[i](outputs[self.output_index[i][j]], gts[:, self.output_index[i][j]], device=device)
                else:
                    loss += self.criteria[i](outputs[self.output_index[i][j]], gts[:, self.output_index[i][j]], device=device)
        return loss
    

class MultiRLLoss(nn.Module):
    def __init__(self, criteria):
        """
        a list of criterions for mutliple outputs, all for reinforcement learning
        
        criteria: list of criterion
        # output_index: list of int, the index of the output that the criterion is applied to
        """
        super().__init__()
        # assert len(criteria) == len(output_index)
        self.criteria = criteria
        # self.output_index = output_index

    def forward(
        self,
        probs,
        values,
        rewards,
        entropys,
        loss_masks=None,
        action_space_masks=None,
        eta=None,
        print_info=False,
        device="cpu",
        old_log_probs=None,
    ):
        """
        probs: for each action_space, (timesteps, batch_size) for PPO or (timesteps, batch_size, action_dim) for A2C
        values: for each action_space, (timesteps, batch_size, 1)
        rewards: (timesteps, batch_size), if there are multiple action spaces applying different rewards, the shape is (timesteps, batch_size, action_num)
        old_log_probs: optional, for PPO only, list of (timesteps, batch_size) tensors
        """
        assert len(probs) == len(values) == len(entropys)
        if old_log_probs is not None:
            assert len(probs) == len(old_log_probs)
        else:
            old_log_probs = [None] * len(probs)
        eta = [None] * len(probs) if eta is None else eta
        # assert len(probs) > max(self.output_index)
        loss, policy_gradient, value_loss, pi_ent = None, None, None, None
        rewards = np.array(rewards)
        for i in range(len(probs)):
            masks = np.logical_and(action_space_masks[:, i], loss_masks)
            for j in range(len(self.criteria)):
                # Check if criterion is PPOLoss (needs old_log_probs) or A2CLoss
                criterion = self.criteria[j]

                if len(rewards.shape) == 3:
                    # each action space has a different reward
                    rewards_i = rewards[:, :, i]
                else:
                    rewards_i = rewards

                if np.any(masks[:rewards_i.shape[0]]) != 0:
                    l, p, v, ent = criterion(
                        probs[i],
                        old_log_probs[i],
                        values[i],
                        rewards_i,
                        entropys[i],
                        masks[:rewards_i.shape[0]],
                        eta[i],
                        print_info,
                        device=device,
                    )

                    if loss is None:
                        loss = l
                        policy_gradient = p
                        value_loss = v
                        pi_ent = ent
                    else:
                        loss += l
                        policy_gradient += p
                        value_loss += v
                        pi_ent += ent
        return loss, policy_gradient, value_loss, pi_ent


class MultiAuxiliaryLoss(nn.Module):
    def __init__(self, criteria):
        """
        a list of criterions for auxiliary loss
        
        criteria: list of criterion
        """
        super().__init__()
        self.criteria = criteria

    def forward(self, device="cpu", **kwargs):
        loss = torch.tensor(0.0, device=device)
        for criterion in self.criteria:
            loss += criterion(device=device, **kwargs)
        return loss
