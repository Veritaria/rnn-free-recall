from math import gamma
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss, mse_loss
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()


class PPOLoss(nn.Module):
    def __init__(self, value_weight=1.0, eta=0.0, gamma=1.0, lam=1.0, clipping_param=0.2,
                 value_loss_func='mse', advantage_normalize=True, returns_normalize=False) -> None:
        """
        compute the objective node for PPO
        """
        super().__init__()
        self.returns_normalize = returns_normalize
        self.value_weight = value_weight
        self.eta = eta
        self.gamma = gamma
        self.lam = lam
        self.clipping_param = clipping_param
        self.value_loss_func = value_loss_func
        self.advantage_normalize = advantage_normalize

    def forward(
        self,
        new_log_probs,
        old_log_probs,
        values,
        rewards,
        entropys,
        loss_masks=None,
        eta=None,
        print_info=False,
        device='cpu',
    ):
        """
        PPO loss (clipped surrogate objective).

        This repo uses the same convention as A2C:
        - new_log_probs: list length T of tensors (batch,)
        - old_log_probs: list length T of tensors (batch,)
        - values: list length T of tensors (batch, 1)
        - rewards: array-like (T, batch)
        - entropys: list length T of tensors (batch,)
        - loss_masks: array-like (T, batch) with 1 for valid timesteps
        """
        eta = self.eta if eta is None else eta

        rewards_arr = np.array(rewards)
        loss_masks_arr = np.array(loss_masks) if loss_masks is not None else np.ones_like(rewards_arr, dtype=bool)

        # (T, B) -> (B, T)
        new_log_probs_bt = torch.stack(new_log_probs).to(device).transpose(1, 0)
        old_log_probs_bt = torch.stack(old_log_probs).to(device).transpose(1, 0)
        values_bt = torch.stack(values).squeeze(2).to(device).transpose(1, 0)
        entropys_bt = torch.stack(entropys).to(device).transpose(1, 0)

        # Compute returns and TD errors
        returns_bt = compute_returns(rewards_arr, loss_masks_arr, gamma=self.gamma, normalize=self.returns_normalize).to(device)
        rewards_bt = torch.tensor(rewards_arr, device=device).transpose(1, 0)
        values_include_last = torch.zeros((values_bt.shape[0], values_bt.shape[1] + 1), device=device)
        values_include_last[:, :-1] = values_bt.detach()
        td_errors = rewards_bt + self.gamma * values_include_last[:, 1:] - values_include_last[:, :-1]
        advantages_bt = compute_advantages(td_errors, loss_masks_arr.T, lam=self.lam, gamma=self.gamma).to(device)

        if self.advantage_normalize:
            adv_mean = advantages_bt.mean(dim=1, keepdim=True)
            adv_std = advantages_bt.std(dim=1, keepdim=True) + eps
            advantages_bt = (advantages_bt - adv_mean) / adv_std

        loss_masks_bt = torch.tensor(loss_masks_arr, device=device).transpose(1, 0)
        valid = loss_masks_bt != 0

        ratio = torch.exp(new_log_probs_bt - old_log_probs_bt)
        ratio_valid = ratio[valid]
        adv_valid = advantages_bt[valid]
        ent_valid = entropys_bt[valid]
        values_valid = values_bt[valid]
        returns_valid = returns_bt[valid]

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(ratio_valid, 1.0 - self.clipping_param, 1.0 + self.clipping_param)
        surrogate1 = ratio_valid * adv_valid
        surrogate2 = clipped_ratio * adv_valid
        policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))

        # Value loss
        if self.value_loss_func == 'mse':
            value_loss = 0.5 * mse_loss(values_valid.float(), returns_valid.float())
        elif self.value_loss_func == 'l1':
            value_loss = smooth_l1_loss(values_valid.float(), returns_valid.float())
        else:
            raise ValueError(f"Unknown value_loss_func: {self.value_loss_func}")

        entropy_term = torch.mean(ent_valid)
        
        total_loss = policy_loss + value_loss * self.value_weight - entropy_term * eta

        if print_info:
            print("PPO loss info (policy, value, entropy):", policy_loss.item(), value_loss.item(), entropy_term.item())

        return total_loss, policy_loss, value_loss, entropy_term * eta


class A2CLoss(nn.Module):
    def __init__(self, returns_normalize=False, use_V=True, value_weight=1.0, eta=0.0, gamma=1.0, lam=1.0, 
                    value_loss_func='mse', advantage_normalize=False) -> None:
        """
        compute the objective node for policy/value networks

        Parameters
        ----------
        probs : list
            action prob at time t
        values : list
            state value at time t
        rewards : list
            rewards at time t

        Returns
        -------
        torch.tensor, torch.tensor
            Description of returned object.

        """
        super().__init__()
        self.returns_normalize = returns_normalize
        self.use_V = use_V
        self.value_weight = value_weight
        self.eta = eta
        self.gamma = gamma
        self.lam = lam
        self.value_loss_func = value_loss_func
        self.advantage_normalize = advantage_normalize
        
    def forward(self, probs, old_log_probs, values, rewards, entropys, loss_masks=None, eta=None, print_info=False, device='cpu'):
        """
        probs: list of torch.tensor, overall size is (timesteps, batch_size)
        old_log_probs: no use, for PPO only
        values: list of torch.tensor, overall size is (timesteps, batch_size, 1)
        rewards, entropys: list of torch.tensor, overall size is (timesteps, batch_size)
        loss_masks: list, overall size is (timesteps, batch_size)
        """
        eta = self.eta if eta is None else eta
        
        rewards, loss_masks = np.array(rewards), np.array(loss_masks)

        # probs: batch_size x timesteps
        # values, returns: batch_size x timesteps
        # entropys: timesteps x batch_size
        probs = torch.stack(probs).to(device).transpose(1, 0)       # (batch_size, timesteps)
        values = torch.stack(values).squeeze(2).to(device).transpose(1, 0)  # (batch_size, timesteps)
        entropys = torch.stack(entropys).to(device).transpose(1, 0)        # (batch_size, timesteps)

        # returns: batch_size x timesteps
        # print(rewards.shape, loss_masks.shape)
        returns = compute_returns(rewards, loss_masks, gamma=self.gamma, normalize=self.returns_normalize)
        rewards = torch.tensor(rewards).to(device).transpose(1, 0)    # (batch_size, timesteps)
        values_include_last = np.zeros((values.shape[0], values.shape[1] + 1))    # values including the last timestep value (default 0 in this codebase)
        values_include_last[:, :-1] = values.data    # (batch_size, timesteps + 1)
        td_errors = rewards + self.gamma * values_include_last[: ,1:] - values_include_last[: ,:-1]    # (batch_size, timesteps)
        advantages = compute_advantages(td_errors, loss_masks.T, lam=self.lam, gamma=self.gamma)    # (batch_size, timesteps)
        if self.advantage_normalize:
            advantages = (advantages - advantages.mean(dim=1).reshape(-1, 1)) / (advantages.std(dim=1).reshape(-1, 1) + eps)
        
        policy_grads, value_losses = [], []

        if loss_masks is None:
            loss_masks = torch.ones_like(returns)
        else:
            loss_masks = torch.tensor(loss_masks).to(device).transpose(1, 0)
        
        batch_size = probs.shape[0]

        if print_info:
            print("loss info:", probs[0], values[0], returns[0])
        if self.use_V:
            # A2C loss
            # print(returns.shape, values.shape)
            # A = returns - values.data
            A = advantages
            if torch.sum(loss_masks) == 0:
                value_losses = torch.tensor(0.0).to(device)
            else:
                if self.value_loss_func == 'mse':
                    value_losses = 0.5 * mse_loss(torch.squeeze(values[loss_masks != 0].to(device).float()), 
                                                torch.squeeze(returns[loss_masks != 0].to(device).float())) * values[loss_masks != 0].shape[0]
                elif self.value_loss_func == 'l1':
                    value_losses = smooth_l1_loss(torch.squeeze(values[loss_masks != 0].to(device).float()), 
                                                torch.squeeze(returns[loss_masks != 0].to(device).float())) * values[loss_masks != 0].shape[0]
            # smooth_l1_loss(torch.squeeze(v_t.to(self.device)), torch.squeeze(R_t.to(self.device)))
        else:
            # policy gradient loss
            A = returns
            value_losses = torch.tensor(0.0).to(device)
        # accumulate policy gradient
        policy_grads = -probs[loss_masks != 0] * A[loss_masks != 0]
        policy_gradient = torch.sum(policy_grads) / batch_size
        value_loss = torch.sum(value_losses) / batch_size
        pi_ent = torch.sum(entropys[loss_masks != 0]) / batch_size
        loss = policy_gradient + value_loss * self.value_weight - pi_ent * self.eta
        return loss, policy_gradient, value_loss, pi_ent * self.eta


def pick_action(action_distribution):
    """action selection by sampling from a multinomial.

    Parameters
    ----------
    action_distribution : a list of 2d torch.tensor, batch_size x action_dim
        each element is an action distribution, pi(a|s) of one of the action spaces

    Returns
    -------
    sampled action: 1d torch.tensor, batch_size    
    log_prob(sampled action): 2d torch.tensor, batch_size x action_dim

    """
    actions, log_prob_actions, actions_max = [], [], []
    for a_d in action_distribution:
        m = Categorical(a_d)
        a_t = m.sample()
        a_t_max = torch.argmax(a_d, dim=1)
        log_prob_a_t = m.log_prob(a_t)
        actions.append(a_t)
        log_prob_actions.append(log_prob_a_t)
        actions_max.append(a_t_max)
    actions = torch.stack(actions)
    log_prob_actions = torch.stack(log_prob_actions)
    actions_max = torch.stack(actions_max)
    return actions, log_prob_actions, actions_max


def compute_returns(rewards, loss_masks, gamma=1.0, normalize=False):
    """
    compute return in the standard policy gradient setting.

    Parameters
    ----------
    rewards : list, 2d array, timestep x batch_size
        immediate reward at time t, for all t
    gamma : float, [0,1]
        temporal discount factor
    normalize : bool
        whether to normalize the return
        - default to false, because we care about absolute scales

    Returns
    -------
    2d torch.tensor, batch_size x timestep
        the sequence of cumulative return

    """
    # compute cumulative discounted reward since t, for all t
    # rewards = np.array(rewards)
    # loss_masks = np.array(loss_masks)
    rewards[loss_masks == 0] = 0

    R = np.zeros(rewards.shape[1])
    returns = np.zeros(rewards.shape)
    for t in range(rewards.shape[0]):
        R = rewards[-t-1] + gamma * R
        returns[-t-1] = R
    if normalize:
        returns = (returns - np.mean(returns, axis=0)) / (np.mean(returns, axis=0) + eps)
    returns = returns.T

    returns[loss_masks.T == 0] = 0
    return torch.tensor(returns)

    # for i in range(rewards.shape[1]):
    #     reward = rewards[:, i]
    #     returns = []
    #     R = 0.0
    #     for r in reward[::-1]:
    #         R = r + gamma * R
    #         returns.insert(0, R)
    #     returns = torch.tensor(returns)
    #     # normalize w.r.t to the statistics of this trajectory
    #     if normalize:
    #         returns = (returns - returns.mean()) / (returns.std() + eps)
    #     returns_all.append(returns)
    # return returns_all


def compute_advantages(td_errors, loss_masks, lam=1.0, gamma=1.0):
    """
    compute the advantage function with generalized advantage estimation.
    code based on https://avandekleut.github.io/a2c/

    Parameters
    ----------
    td_errors: 2d torch.tensor, batch_size x timesteps
        the sequence of td errors
    loss_masks: 2d torch.tensor, batch_size x timesteps
        the sequence of loss masks
    lam: float, [0,1]
        the lambda parameter for the advantage function
        1.0 means using the returns, 0.0 means using the td errors
    gamma: float, [0,1]
        the gamma parameter for the advantage function

    Returns
    -------
    2d torch.tensor, batch_size x timesteps
        the sequence of advantages
    """
    advantages = np.zeros_like(td_errors)
    advantages[:,-1] = td_errors[:,-1]
    for t in range(td_errors.shape[1] - 2, -1, -1):
        advantages[:,t] = td_errors[:,t] + gamma * lam * advantages[:,t+1]
    advantages[loss_masks == 0] = 0
    return torch.tensor(advantages)
