"""
Metrics and losses for network training
"""
import torch

def L2_loss(batch):
    """Given a batch of latents (shaped batch,sequence,activations), calculates the squared norm of the activations across the entire batch and returns shape batch. 

    """
    squared_acts = batch**2
    l2_norm = torch.sum(squared_acts,axis=-1)
    mean_l2 = torch.mean(l2_norm)
    return mean_l2
