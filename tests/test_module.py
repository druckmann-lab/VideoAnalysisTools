"""
"""
from behavioral_autoencoder.module import SingleSessionModule
from behavioral_autoencoder.dataset import CropResizeProportion
from behavioral_autoencoder.dataloading import SessionFramesDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import cv2
import os 
import json

here = os.path.join(os.path.abspath(os.path.dirname(__file__)))

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

class Test_SingleSessionModule():
    model_config_path = os.path.join(here,"..","configs","model_configs","alm_default.json")
    train_config_path = os.path.join(here,"..","configs","train_configs","alm_default.json")
    crop_config_path = os.path.join(here,"..","configs","data_configs","alm_side.json")
    def test_init(self):
        with open(self.model_config_path,"r") as f:
            model_config = json.load(f)
        with open(self.train_config_path,"r") as f:
            train_config = json.load(f)
        hparams = {
                "model":"single_session_autoencoder",
                "model_config":model_config,
                "train_config":train_config
                }
        ssm = SingleSessionModule(hparams)
    def test_baby_train_loop(self,tmp_path):
        model_config_path = os.path.join(here,"..","configs","model_configs","alm_default.json")
        train_config_path = os.path.join(here,"..","configs","train_configs","alm_default.json")

        with open(self.model_config_path,"r") as f:
            model_config = json.load(f)
        with open(self.train_config_path,"r") as f:
            train_config = json.load(f)
        with open(self.crop_config_path,"r") as f:
            crop_config = json.load(f)

        hparams = {
                "model":"single_session_autoencoder",
                "model_config":model_config,
                "train_config":train_config
                }
        ssm = SingleSessionModule(hparams)

        session_dir = temp_hierarchical_folder_generator(
            tmp_path,
        )

        alm_cropping = CropResizeProportion(self.crop_config_path)
        data_config = {
                "data_path":session_dir,
                "transform":alm_cropping,
                "extension":".png",
                "trial_pattern":None
                }
        sfdm = SessionFramesDataModule(data_config,10,2,10,1,10,1)
        logger = TensorBoardLogger("tb_logs",name="test_single_session_auto",log_graph=True)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator='cpu',  # Explicitly use CPU for testing
            enable_checkpointing=False,  # Disable for testing
            logger=logger,  # Disable logging for testing
            enable_progress_bar=True,  # See training progress
        )
        trainer.fit(ssm,sfdm)
        assert trainer.current_epoch == 1, "Training should complete 2 epochs"
        assert trainer.global_step > 0, "Should have completed some training steps"
    def test_eval_save(self,tmp_path):
        model_config_path = os.path.join(here,"..","configs","model_configs","alm_default.json")
        train_config_path = os.path.join(here,"..","configs","train_configs","alm_default.json")

        with open(self.model_config_path,"r") as f:
            model_config = json.load(f)
        with open(self.train_config_path,"r") as f:
            train_config = json.load(f)
        with open(self.crop_config_path,"r") as f:
            crop_config = json.load(f)

        hparams = {
                "model":"single_session_autoencoder",
                "model_config":model_config,
                "train_config":train_config
                }
        ssm = SingleSessionModule(hparams)

        session_dir = temp_hierarchical_folder_generator(
            tmp_path,
        )

        alm_cropping = CropResizeProportion(self.crop_config_path)
        data_config = {
                "data_path":session_dir,
                "transform":alm_cropping,
                "extension":".png",
                "trial_pattern":None
                }
        sfdm = SessionFramesDataModule(data_config,10,2,10,1,10,1)
        logger = TensorBoardLogger("tb_logs",name="test_single_session_auto",log_graph=True)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator='cpu',  # Explicitly use CPU for testing
            enable_checkpointing=False,  # Disable for testing
            logger=logger,  # Disable logging for testing
            enable_progress_bar=True,  # See training progress
        )
        trainer.fit(ssm,sfdm)
        # Extract out some latents: 
        full_dataloader = DataLoader(sfdm.dataset,batch_size=10)
        predictions = []
        latents = []
        for batch in full_dataloader:
            prediction,latent = ssm(batch)
            predictions.append()
        import pdb; pdb.set_trace()    

