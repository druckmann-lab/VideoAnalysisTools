"""
Functions for two kinds of evaluation:
    1. Bulk extraction of latents and predictions from the last timepoint. 
    2. Evaluation of the latent space. 
"""
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def get_all_predicts_latents(trained_model,datamodule,batch_size,num_workers):
    """
    Get latents for the entire video, having trained on a subsampled set. 

    """
    full_dataloader = DataLoader(datamodule.dataset,batch_size=batch_size,num_workers=num_workers)
    predictions = []
    latents = []
    for batch in tqdm(full_dataloader):
        prediction,latent = trained_model(batch)
        predictions.append(prediction)
        latents.append(latent)
    return torch.concatenate(predictions), torch.concatenate(latents)    

def get_dl_predicts_latents(trained_model,dataloader,batch_size,num_workers):
    """
    Get latents for the entire video, having trained on a subsampled set. 

    """
    predictions = []
    latents = []
    for batch in tqdm(dataloader):
        prediction,latent = trained_model(batch)
        predictions.append(prediction)
        latents.append(latent)
    return torch.concatenate(predictions), torch.concatenate(latents)    
