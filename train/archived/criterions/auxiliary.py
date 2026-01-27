import numpy as np
import torch
import torch.nn as nn
from models.utils import entropy


class MemoryOneHotRegularization(nn.Module):
    def __init__(self, weight=0.001):
        super().__init__()
        self.weight = weight

    def forward(self, device="cpu", **kwargs):
        """
        memory: torch.tensor, size is (timesteps, batch_size, memory_dim)
        """
        memory_similarity = kwargs['memory_similarity']
        batch_size = memory_similarity.shape[1]
        memory_similarity = memory_similarity.reshape(-1, memory_similarity.shape[-1])
        sim_entropy = entropy(memory_similarity, device)
        # sim_entropy = torch.stack(sim_entropy)
        loss = torch.sum(sim_entropy) / batch_size * self.weight
        return loss
