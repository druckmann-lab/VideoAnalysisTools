import torchvision
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Subset,DataLoader
from behavioral_autoencoder.dataset import SessionFramesTorchvision,SessionSequenceTorchvision,CropResizeProportion
import pytorch_lightning as pl
from joblib import Memory
import os
import tempfile
from pathlib import Path

# Set up cache location with multiple options for flexibility
def get_cache_dir():
    """Get cache directory with the following priority:
    1. BEHAVIORAL_AUTOENCODER_CACHE env variable if set
    2. Project's .cache directory if in development mode
    3. User's home directory under ~/.cache/behavioral_autoencoder
    4. System temp directory as fallback
    """
    # Option 1: Environment variable (highest priority)
    if "BEHAVIORAL_AUTOENCODER_CACHE" in os.environ:
        cache_dir = Path(os.environ["BEHAVIORAL_AUTOENCODER_CACHE"])
    
    # Option 2: Project directory if it exists (development mode)
    elif (Path(__file__).parent.parent.parent / ".cache").exists():
        cache_dir = Path(__file__).parent.parent.parent / ".cache"
    
    # Option 3: User's home directory
    else:
        home_dir = Path.home()
        cache_dir = home_dir / ".cache" / "behavioral_autoencoder"
    
    # Create directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

# Initialize memory cache
memory = Memory(location=get_cache_dir(), verbose=1)

@memory.cache
def calculate_mean_image(data_path, dataset_config, subsample_rate, subsample_offset, batch_size, num_workers):
    """
    Calculate mean image from given training set parameters. 
    This function is cached to avoid redundant calculations.
    """
    # 1. First construct the right training set indices: 
    sub_dataset = {}
    for field in ["transform","extension","trial_pattern"]:
        sub_dataset[field] = dataset_config[field]
    dataset = SessionFramesTorchvision(data_path,**sub_dataset)
    all_indices = np.arange(len(dataset))
    ## Subsample indices
    train_inds = all_indices[subsample_offset::subsample_rate]
    trainset = Subset(dataset, train_inds)

    ## use dataloader to compute sum image 
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)

    sum_im = torch.zeros(dataset[0].shape[1:]) ## should be 
    for data in tqdm(trainloader):
        sum_im += data.sum(axis=0).sum(axis=0) ## sum across the batch and sequence dimensions.
    mean = sum_im / len(trainset)    
    return mean

class SessionFramesDataModule(pl.LightningDataModule):
    """Lightning data module collects together data loading logic frame subsampling, and then calculates the framewise mean of the training dataset. 
    This datamodule does the following: 
        1. Subsamples at a given subsetting rate to generate the training data.  

        2. Checks if we have already calculated the mean image for the given training parameters 
        3. Generates the training dataset by subsampling at the given subsampling rate and offset. Calculates mean image if does not exist yet. 
        4. Defines datasets which compose given transformations with a subtraction of the training set mean. 

    """
    def __init__(
        self,
        dataset_config,
        batch_size,
        num_workers,
        train_subsample_rate,
        train_subsample_offset,
        val_subsample_rate,
        val_subsample_offset,
        trainset_subtract_mean=True,
        ):
        """
        By default, checks if we have already calculated the image mean on a particular dataset for a particular training set, and if so gets that cached mean.

        Parameters
        ----------
        dataset_config : dict
            dictionary containing configuration parameters for dataset. Must include:
                data_path: str
                    path of the top level folder of per-trial frames.
                transform : any
                    the transform function that we will apply to all the data. Can be followed by different training, testing transforms for different datasets, and can be None.
            Can optionally also include:
                extension : str
                    extension of files which
                trial_pattern : str
                    regex which filters for certain folder prefixes in the trial.
        batch_size : int
            frames per batch
        trainset_subsample_rate : int
            take one frame for every `trainset_subsample_rate` frames.
        trainset_subsample_offset : int
            offset for subsampling.
        trainset_subtract_mean : bool
            whether we should calculate and subtract the mean of the training set.
        """
        super().__init__()
        self.data_path = dataset_config.pop("data_path") ## each dataset does not take this argument, so we remove. 
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_subsample_rate = train_subsample_rate
        self.train_subsample_offset = train_subsample_offset
        self.val_subsample_rate = val_subsample_rate
        self.val_subsample_offset = val_subsample_offset
        self.subtract_mean = trainset_subtract_mean

        if self.subtract_mean:
            print("Calculating mean image")
            self.mean_image = calculate_mean_image(self.data_path,dataset_config,self.train_subsample_rate,self.train_subsample_offset,self.batch_size,self.num_workers)
            subtract_mean = torchvision.transforms.Lambda(self.subtract_mean_image)
            # augment the transformation for our datasets: 
            if self.dataset_config["transform"] is not None:
                mean_normed_transform = torchvision.transforms.Compose([
                    self.dataset_config["transform"],
                    subtract_mean
                    ])
            else:
                mean_normed_transform = torchvision.transforms.Compose([
                    subtract_mean
                    ])
            self.transform = mean_normed_transform
        else:
            self.transform = self.dataset_config["transform"]
            print("Done with setup")

    def subtract_mean_image(self,x):
        return x-self.mean_image

    def setup(self,stage):
        if stage == "fit":
            _ = self.dataset_config.pop("transform")
            self.dataset = SessionFramesTorchvision(self.data_path,transform = self.transform,**self.dataset_config)
            all_indices = np.arange(len(self.dataset))
            ## Subsample indices
            train_inds = all_indices[self.train_subsample_offset::self.train_subsample_rate]
            val_inds = all_indices[self.val_subsample_offset::self.val_subsample_rate]
            #test_inds = [i for i in all_indices if not (i in train_inds)]
            self.trainset = Subset(self.dataset,train_inds)
            self.valset = Subset(self.dataset,val_inds)
        if stage == "test":    
            ### the only way to access  this right now is to pass setup explicitly right now. 
            _ = self.dataset_config.pop("transform")
            self.dataset = SessionSequenceTorchvision(self.data_path,transform = self.transform,**self.dataset_config)

    def train_dataloader(self,shuffle=True):
        dataloader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last =False,
            pin_memory=True
                )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last =False,
            pin_memory=True
                )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

