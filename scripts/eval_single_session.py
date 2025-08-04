"""
Evaluate a single session autoencoder. 

Assumes that we are given a path to a directory `preds/{modeltype}/date/time/`. Will use the configuration parameters stored there 
"""
import os
import datetime
from tqdm import tqdm
import json
import numpy as np
import fire
import torch
import pytorch_lightning as pl
from behavioral_autoencoder.module import SingleSessionModule
from behavioral_autoencoder.dataloading import SessionFramesDataModule
from behavioral_autoencoder.dataset import CropResizeProportion

here = os.path.join(os.path.abspath(os.path.dirname(__file__)))

def eval_trialwise(model,test_dataset,mean_image,path):
    """
    """
    for i, image_sequence in tqdm(enumerate(test_dataset)):
        reconstructs,latents = model(image_sequence[None,:].cuda())
        reconstructs_centered = reconstructs + mean_image.cuda()
        folder = test_dataset.trial_folders[i]
        savepath = os.path.join(path,folder)
        try:
            os.mkdir(savepath)
        except FileExistsError:    
            pass
        np.save(os.path.join(savepath,"reconstruct.npy"),reconstructs_centered.cpu().detach().numpy())
        np.save(os.path.join(savepath,"latents.npy"),latents.cpu().detach().numpy())

def main(data_path,data_config_path,eval_config_path):
    """
    get a model path, and use it to load in a given model. 
    """
    saved_checkpoint_path = os.path.join(".","models","single_session_autoencoder","03-07-25","18_05_06","epoch=99-step=33100.ckpt")
    data_dir = os.path.join("home","ubuntu","Data","CW35","2023_12_15","Frames")
    metadata_dir = os.path.join(".","preds","single_session_autoencoder","03-07-25","18_05_06")
    video_fps = 400
    delay_start_time = 2.5+1.3 ## pre-sample and sample time intervals. 
    delay_end_time = 2.5+1.3+3 ## delay is 3 seconds.
    subsample = 10
    eval_batch_size=1

    ## Load in data related stuff

    with open(data_config_path,"r") as f:
        data_process_config = json.load(f)

    with open(eval_config_path,"r") as f: 
        eval_config = json.load(f)

    alm_cropping = CropResizeProportion(data_config_path)
    data_config = {
            "data_path":data_path,
            "transform":alm_cropping,
            "extension":data_process_config["extension"],
            "trial_pattern":data_process_config["trial_pattern"],
            "frame_subset":[f"frame_{i:06d}.png" for i in np.arange(int(delay_start_time*video_fps),int(delay_end_time*video_fps),subsample)]
            }

    date = datetime.datetime.now().strftime("%m-%d-%y")
    time = datetime.datetime.now().strftime("%H_%M_%S")
    datestamp_eval = os.path.join(here,"eval",date)
    timestamp_eval = os.path.join(here,"eval",date,time)
    for path in [datestamp_eval,timestamp_eval]:
        try:
            os.mkdir(path)
        except FileExistsError:    
            pass

    sfdm = SessionFramesDataModule(
            data_config,
            eval_config["batch_size"],
            eval_config["num_workers"],
            eval_config["subsample_rate"],
            eval_config["subsample_offset"],
            eval_config["val_subsample_rate"],
            eval_config["val_subsample_offset"]
            )

    model = SingleSessionModule.load_from_checkpoint(
            checkpoint_path=saved_checkpoint_path
            )

    sfdm.setup("test")

    eval_trialwise(model,sfdm.dataset,sfdm.mean_image,path)
        

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    fire.Fire(main)

