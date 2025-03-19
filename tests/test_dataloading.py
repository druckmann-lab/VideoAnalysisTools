"""
Test what the dataloader does. 
"""
from behavioral_autoencoder.dataset import CropResizeProportion
import json
import os
import cv2
import numpy as np
from behavioral_autoencoder.dataloading import SessionFramesDataModule


here = os.path.abspath(os.path.dirname(__file__))

def temp_hierarchical_folder_generator(tmp_path, n_trials=3, n_ims_per_trial=10, extra_files=None):
    """Creates a temporary hierarchical folder structure for testing.
    
    This fixture creates a folder structure that mimics a session with multiple trials,
    where each trial contains randomly sampled images. The structure is:
    temp_session/
        ├── 0_trial/
        │   ├── frame_000000.png
        │   ├── frame_000001.png
        │   ├── ...
        │   └── extra_file.txt
        ├── 1_trial/
        │   ├── frame_000000.png
        │   ├── ...
        │   └── extra_file.txt
        └── ...
    
    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing temporary directory path
    n_trials : int, optional
        Number of trial folders to create, by default 3
    n_ims_per_trial : int, optional
        Number of images to create per trial, by default 10
    extra_files : list of str, optional
        List of extra file names to create in each trial directory
        
    Returns
    -------
    Path
        Path to the created temporary session directory
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load example images
    example_images = np.load('./test_data/example_images.npy')*255
    
    # Create session directory
    session_dir = tmp_path / "temp_session"
    session_dir.mkdir()
    
    # Create trial folders and populate with images
    for trial in range(n_trials):
        trial_dir = session_dir / f"{trial}_trial"
        trial_dir.mkdir()
        
        # Randomly sample images
        selected_images = example_images[
            np.random.choice(len(example_images), n_ims_per_trial, replace=True)
        ]
        
        # Save images as PNGs
        for i, img in enumerate(selected_images):
            img_path = trial_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(img_path), img)
        
        # Create extra files within each trial folder if specified
        if extra_files:
            for filename in extra_files:
                (trial_dir / filename).touch()
    
    return session_dir

class Test_SessionFramesDataModule():
    config_path = os.path.join(here,"..","configs","data_configs","alm_side.json")
    def test_init(self,tmp_path):
        with open(self.config_path,"r") as f:
            crop_config = json.load(f)
        # Create a larger dataset
        n_trials = 10  # Increased number of trials
        n_ims_per_trial = 50  # Increased images per trial
        session_dir = temp_hierarchical_folder_generator(
            tmp_path,
            n_trials=n_trials,
            n_ims_per_trial=n_ims_per_trial
        )
        
        alm_cropping = CropResizeProportion(self.config_path)
        data_config = {
                "data_path":session_dir,
                "transform":alm_cropping,
                "extension":".png",
                "trial_pattern":None
                }
        sfdm = SessionFramesDataModule(data_config,10,2,10,1,10,1)
        assert sfdm.mean_image.shape == (1,crop_config["target_h"],crop_config["target_w"])

    def test_train_dataloader(self,tmp_path):
        with open(self.config_path,"r") as f:
            crop_config = json.load(f)
        # Create a larger dataset
        n_trials = 10  # Increased number of trials
        n_ims_per_trial = 50  # Increased images per trial
        session_dir = temp_hierarchical_folder_generator(
            tmp_path,
            n_trials=n_trials,
            n_ims_per_trial=n_ims_per_trial
        )
        
        alm_cropping = CropResizeProportion(self.config_path)
        data_config = {
                "data_path":session_dir,
                "transform":alm_cropping,
                "extension":".png",
                "trial_pattern":None
                }
        sfdm = SessionFramesDataModule(data_config,10,2,10,1,10,1)
        sfdm.setup("fit")
        dataloader = sfdm.train_dataloader()
        
        for batch in dataloader:
            assert batch.shape[1:] == (1,1,crop_config["target_h"],crop_config["target_w"])
