"""Build out pytorch datasets for different formatting for behavioral videos.
1. A directory containing videos corresponding to individual trials.
2. A tar file.
3. TODO: A directory of videos with DALI/NVVL.
    - https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.readers.video_resize.html
    - https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-lightning.html
"""
import numpy as np
import cv2
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms.functional as F
import os
import json
from torch.utils.data import Dataset,DataLoader

import torch
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
import re  # Added for regex pattern matching


## Reference preprocessing functions
def transform_image(image_path, target_shape = (120,112,1), crop_info = {'h_coord': 26}):
    """
    Given a path to an image file, loads it as a grayscale image and transforms it to the given target shape, given some cropping information in the target aspect ratio space.

    Parameters
    ----------
    image_path : str
        string to image
    target_shape : tuple
        tuple giving the target image shape. x and y only are used.
    crop_info: dict
        cropping information to be passed to `transform_image`, with one expected key, `h_coord`. Crops out the image in the original space so that ~h_coord pixels to the right of the image would be cropped following appropriate image transformation.
    """
    img = cv2.imread(image_path,0)
    h_origin, w_origin = img.shape
    h, w, c = target_shape
    img = img[int(crop_info['h_coord'] / h * h_origin + 0.5):,:]
    img = cv2.resize(img,(w,h))
    img = (img / 255.).astype(np.float32)
    return img

def transform_image_from_tar(tar_obj, member_name, target_shape=(120, 112, 1), crop_info={'h_coord': 26}):
    """
    Extracts an image drectoy from the tar archive and applies the same transforms as in `transform_image`.

    Parameters
    ----------
    tar_obj : tar object
        (output of tarfile.open)
    member_name : str
        the relative path to the frame within the tar object.
    target_shape : tuple
        tuple giving the target image shape. x and y only are used.
    crop_info: dict
        cropping information to be passed to `transform_image`, with one expected key, `h_coord`. Crops out the image in the original space so that ~h_coord pixels to the right of the image would be cropped following appropriate image transformation.
    """
    # Extract the file-like object from the tar
    fileobj = tar_obj.extractfile(member_name)
    if fileobj is None:
        raise ValueError(f"Could not extract {member_name}")
    # Read the raw bytes and convert them into a numpy array
    file_bytes = fileobj.read()
    arr = np.frombuffer(file_bytes, np.uint8)
    # Decode the image (read in grayscale mode)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    h_origin, w_origin = img.shape
    h, w, c = target_shape
    # Crop the image based on the provided crop_info
    img = img[int(crop_info['h_coord'] / h * h_origin + 0.5):, :]
    # Resize and normalize the image
    img = cv2.resize(img, (w, h))
    img = (img / 255.).astype(np.float32)
    return img

class SessionFramesDataset(Dataset):
    """
    Assumes we have a dataset which is organized as a directory of directories, with one directory per trial.
    Each trial directory contains frames corresponding to images within the dataset.
    Assumes that video frames are named with some sort of convention "same_path_within_trial_{frame_number}.png"

    Attributes
    ----------
    base_folder : str
        given by parameter at initialization
    crop_info : dict    
        given by parameter at initialization
    trial_folders : arraylike    
        sorted names of per-trial directories 
    extension : str    
        extension for frame files.
    frame_dict : dict    
        dictionary with keys given by trial_folder names, and entries arraylikes of frames within each folder.  
    trial_lengths : list    
        number of frames within each trial dictionary
    cumsum_n_trials : arraylike    
        cumulative index for frame index across all trials. 

    """

    def __init__(self, base_folder, extension=".png", crop_info={'h_coord': 26}, trial_pattern=None):
        """
        Parameters
        ----------
        base_folder : string
            path to the base folder which contains folders for each individual trial. 
        extension : string
            file extension for frame files (default: ".png")
        crop_info : dict    
            cropping information to be passed to `transform_image`, with one expected key, `h_coord`. 
            Crops out the image in the original space so that ~h_coord pixels to the right of the 
            image would be cropped following appropriate image transformation. 
        trial_pattern : string, optional
            Regular expression pattern to match trial folders. If None, all directories are considered
            trial folders. Example: r"^\d+_trial$" would match folders like "0_trial", "1_trial", etc.
        """
        self.base_folder = base_folder
        self.crop_info = crop_info
        self.extension = extension
        
        # Get all items in the base folder
        all_items = os.listdir(base_folder)
        
        # Filter for directories only
        self.trial_folders = [
            item for item in all_items 
            if os.path.isdir(os.path.join(base_folder, item))
        ]
        
        # Apply regex pattern if provided
        if trial_pattern is not None:
            pattern = re.compile(trial_pattern)
            self.trial_folders = [
                folder for folder in self.trial_folders 
                if pattern.match(folder)
            ]
        
        self.trial_folders = np.sort(self.trial_folders)
        
        self.frame_dict = {folder: np.sort(self.filter_frames(base_folder,folder)) for folder in self.trial_folders}
        self.trial_lengths = [len(self.frame_dict[folder]) for folder in self.trial_folders]
        self.cumsum_n_trials = np.cumsum(self.trial_lengths)

    def filter_frames(self,base_folder,folder):
        """
        Given a trial folder, filters out extra files to return only those with a given extension. 
        """
        candidates = os.listdir(os.path.join(base_folder,folder))
        return [f for f in candidates if f.endswith(self.extension)]

    def __len__(self):
        """
        Required method for pytorch datasets. 
        """
        return np.sum(self.trial_lengths)

    def __getitem__(self, idx, method = "searchsorted"):
        """
        TODO: check if argmax affects performance at dataloading. 

        Parameters 
        ----------
        idx: int
            integer index into the data. 
        """
        ## get trial number
        if method == "argmax":
            trial_idx = np.argmax(self.cumsum_n_trials > idx)
        elif method == "searchsorted":    
            trial_idx = np.searchsorted(self.cumsum_n_trials, idx, side='right')

        ## get frame number
        if trial_idx > 0:
            frame_idx = idx - self.cumsum_n_trials[trial_idx - 1]
        else:
            frame_idx = idx

        img = transform_image(
            image_path = os.path.join(
                self.base_folder,
                self.trial_folders[trial_idx],
                self.frame_dict[self.trial_folders[trial_idx]][frame_idx]), 
            crop_info = self.crop_info)
        return img

class CustomCropResize:
    """
    For use with WebDataset to load images in directly from tarball. 

    """
    def __init__(self, target_shape=(120, 112, 1), crop_info={'h_coord': 26}):
        """
        Parameters
        ----------

        target_shape: array 
            shape (height, width, channels)
        crop_info: dict 
            with cropping parameters. For example,
                   crop_info['h_coord'] determines the crop start as:
                   int(crop_info['h_coord'] / target_height * original_height + 0.5)
        """
        self.target_shape = target_shape
        self.crop_info = crop_info

    def __call__(self, img):
        # Ensure the image is in grayscale
        if img.mode != 'L':
            img = img.convert('L')
        # Get target height and width from target_shape
        h_target, w_target, _ = self.target_shape
        # Get original dimensions (PIL gives (width, height))
        original_width, original_height = img.size
        # Compute the vertical crop coordinate, analogous to your cv2 code
        crop_y = int(self.crop_info['h_coord'] / h_target * original_height + 0.5)
        # Crop the image: from crop_y to bottom, full width
        img = F.crop(img, crop_y, 0, original_height - crop_y, original_width)
        # Resize the image to the target dimensions.
        # F.resize expects size as (height, width).
        img = F.resize(img, (h_target, w_target))
        # Convert the image to a tensor (this scales pixel values to [0, 1])
        img = F.to_tensor(img)
        return img

## Preprocessing functions

class CropResizeProportion:
    """Applies a deterministic crop and resize as a standard preprocessing step for the autoencoder. Parameters are given in a configuration file. 
    """
    def __init__(self, cropresizeconfig):
        """if given, proporitonal_{h/w}_coord_{top/bottom} will give proportions by which to remove space along height or width dimensions. 
        target_h,target_w must be given. 

        """
        with open(cropresizeconfig,"r") as f:
            config = json.load(f)
        self.proportional_h_coord_top = config.get("proportional_h_coord_top",None)    
        self.proportional_h_coord_bottom = config.get("proportional_h_coord_bottom",None)    
        self.proportional_w_coord_left = config.get("proportional_w_coord_left",None)    
        self.proportional_w_coord_right = config.get("proportional_w_coord_right",None)    
        self.target_h = config.get("target_h")
        self.target_w = config.get("target_w")

    def __call__(self,img):    
        # Ensure the image is in grayscale
        if img.mode != 'L':
            img = img.convert('L')
        img = self.crop_img_proportional(img)
        # Resize the image to the target dimensions.
        # F.resize expects size as (height, width).
        img = F.resize(img, (self.target_h, self.target_w))
        # Convert the image to a tensor (this scales pixel values to [0, 1])
        img = F.to_tensor(img)
        return img

    def crop_img_proportional(self,img):
        """If given, do proportional crops along each dimension
        """
        # Get original dimensions (PIL gives (width, height))
        original_width, original_height = img.size
        if self.proportional_h_coord_top:
            y_top = int(self.proportional_h_coord_top*original_height+0.5)
        else:     
            y_top=0
        if self.proportional_h_coord_bottom:    
            y_bottom = int(self.proportional_h_coord_bottom*original_height-0.5)
        else:    
            y_bottom = original_height
        if self.proportional_w_coord_left:
            x_left= int(self.proportional_w_coord_left*original_width+0.5)
        else:    
            x_left=0
        if self.proportional_w_coord_right:    
            x_right= int(self.proportional_w_coord_right*original_width-0.5)
        else:    
            x_right=original_width
        # Crop the image: from crop_y to bottom, full width
        img = F.crop(img, y_top, x_left, y_bottom - y_top, x_right-x_left)
        return img

class SessionFramesTorchvision(Dataset): 
    """Essentially the same as SessionFramesDataset above, but factors out image transformations into a separate class. 
    Assumes we have a dataset which is organized as a directory of directories, with one directory per trial.
    Each trial directory contains frames corresponding to images within the dataset.
    Assumes that video frames are named with some sort of convention "same_path_within_trial_{frame_number}.png"

    Attributes
    ----------
    base_folder : str
        given by parameter at initialization
    trial_folders : arraylike    
        sorted names of per-trial directories 
    extension : str    
        extension for frame files.
    frame_dict : dict    
        dictionary with keys given by trial_folder names, and entries arraylikes of frames within each folder.  
    trial_lengths : list    
        number of frames within each trial dictionary
    cumsum_n_trials : arraylike    
        cumulative index for frame index across all trials. 
    transform : any    
        None or transform function 
    """

    def __init__(self, base_folder, extension=".png", trial_pattern=None, transform = None):
        """
        Parameters
        ----------
        base_folder : string
            path to the base folder which contains folders for each individual trial. 
        extension : string
            file extension for frame files (default: ".png")
        crop_info : dict    
            cropping information to be passed to `transform_image`, with one expected key, `h_coord`. 
            Crops out the image in the original space so that ~h_coord pixels to the right of the 
            image would be cropped following appropriate image transformation. 
        trial_pattern : string, optional
            Regular expression pattern to match trial folders. If None, all directories are considered
            trial folders. Example: r"^\d+_trial$" would match folders like "0_trial", "1_trial", etc.
        """
        self.base_folder = base_folder
        self.extension = extension
        
        # Get all items in the base folder
        all_items = os.listdir(base_folder)
        
        # Filter for directories only
        self.trial_folders = [
            item for item in all_items 
            if os.path.isdir(os.path.join(base_folder, item))
        ]
        
        # Apply regex pattern if provided
        if trial_pattern is not None:
            pattern = re.compile(trial_pattern)
            self.trial_folders = [
                folder for folder in self.trial_folders 
                if pattern.match(folder)
            ]
        
        self.trial_folders = np.sort(self.trial_folders)
        
        self.frame_dict = {folder: np.sort(self.filter_frames(base_folder,folder)) for folder in self.trial_folders}
        self.trial_lengths = [len(self.frame_dict[folder]) for folder in self.trial_folders]
        self.cumsum_n_trials = np.cumsum(self.trial_lengths)
        self.transform = transform

    def filter_frames(self,base_folder,folder):
        """
        Given a trial folder, filters out extra files to return only those with a given extension. 
        """
        candidates = os.listdir(os.path.join(base_folder,folder))
        return [f for f in candidates if f.endswith(self.extension)]

    def __len__(self):
        """
        Required method for pytorch datasets. 
        """
        return np.sum(self.trial_lengths)

    def __getitem__(self, idx, method = "searchsorted"):
        """
        TODO: check if argmax affects performance at dataloading. 

        Parameters 
        ----------
        idx: int
            integer index into the data. 
        """
        ## get trial number
        if method == "argmax":
            trial_idx = np.argmax(self.cumsum_n_trials > idx)
        elif method == "searchsorted":    
            trial_idx = np.searchsorted(self.cumsum_n_trials, idx, side='right')

        ## get frame number
        if trial_idx > 0:
            frame_idx = idx - self.cumsum_n_trials[trial_idx - 1]
        else:
            frame_idx = idx

        image_path = os.path.join(
            self.base_folder,
            self.trial_folders[trial_idx],
            self.frame_dict[self.trial_folders[trial_idx]][frame_idx])

        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        return img

class SessionMetadataHandler:
    def __init__(self, config, mode='local', animal=None, session=None):
        """
        Initializes the handler with a configuration dictionary, running mode ('local' or 'aws'),
        and session identifiers.
        """
        self.config = config
        self.mode = mode
        self.animal = animal
        self.session = session
        
        # State variables
        self.bpod_data = None
        self.behavior_df = None
        self.filtered_behavior_df = None
        self.video_train_test_split_df = None
        
        # Store paths
        self.bpod_path = f"{self.config['bpod_folder'][self.mode]}{self.animal}/{self.session}.bpod.npy"
        self.h5_path = f"{self.config['h5_folder'][self.mode]}{self.session}/{self.session}_side.h5"

    def load_files(self):
        """Loads bpod dataset lazily from disk."""
        self.bpod_data = np.load(self.bpod_path, allow_pickle=True)

    def build_behavior_df(self):
        """Extracts the behavior trials and aligns them with video trials."""
        behavior_array = self.bpod_data[0]
        video_files_list = self.bpod_data[2]
        
        video_trial_ids = []
        for video_file in video_files_list:
            if len(video_file[0]) > 0:
                video_trial_ids.append(int(video_file[0].split('_trial')[1].split('_date')[0]))
            else:
                video_trial_ids.append(None)
        
        video_trial_ids = pd.Series(video_trial_ids, dtype='Int64')
        behavior_map = self.config['behavior_channel_map']
        
        self.behavior_df = pd.DataFrame({
            name: behavior_array[idx, :] for name, idx in behavior_map.items()
        })
        self.behavior_df['trial_id'] = self.behavior_df.index
        self.behavior_df['video_trial_id'] = video_trial_ids

    def filter_trials(self):
        """Determines the availability of videos in H5 and applies boolean filtering rules."""
        # Open the H5 file temporarily to read the trial IDs
        with h5py.File(self.h5_path, 'r') as h5_dataset:
            available_video_trials = np.unique(h5_dataset['trial_ids'])
            
        self.behavior_df['video_available'] = self.behavior_df['video_trial_id'].isin(available_video_trials)
        
        df = self.behavior_df
        # You could also move these column filters into the config in the future
        mask = (df['protocol'] == 5) & \
               (df['outcome'] < 3) & \
               (df['early_lick_sample'] == 0) & \
               (df['early_lick_delay'] == 0) & \
               (df['autowater'] == 0)
               
        self.filtered_behavior_df = df[mask & df['video_available']].copy()

    def train_test_split(self):
        """Applies train/test splitting conditionally on the chosen columns."""
        strat_cols = self.config.get('stratification_columns', ['outcome', 'trial_type', 'airpuff'])
        seed = self.config.get('split_random_seed', 42)
        
        train_idx, test_idx = train_test_split(
            self.filtered_behavior_df.index, 
            test_size=0.5, 
            random_state=seed, 
            stratify=self.filtered_behavior_df[strat_cols]
        )
        
        self.filtered_behavior_df['train_split'] = 0
        self.filtered_behavior_df.loc[train_idx, 'train_split'] = 1
        self.filtered_behavior_df.loc[test_idx, 'train_split'] = 2

    def get_video_split_df(self):
        """Isolates the necessary final output columns."""
        strat_cols = self.config.get('stratification_columns', ['outcome', 'trial_type', 'airpuff'])
        cols_to_keep = strat_cols + ['train_split', 'video_trial_id', 'trial_id']
        self.video_train_test_split_df = self.filtered_behavior_df[cols_to_keep]
        return self.video_train_test_split_df
        
    def process_all(self):
        """Convenience method to execute all steps sequentially."""
        self.load_files()
        self.build_behavior_df()
        self.filter_trials()
        self.train_test_split()
        return self.get_video_split_df()


class H5VideoDataset(Dataset):
    def __init__(self, h5_path, valid_trials_df, split='train', config=None):
        """
        Args:
            h5_path (str): Path to the HDF5 file.
            valid_trials_df (pd.DataFrame): DataFrame containing valid trials (with train_split, video_trial_id).
            split (str): 'train', 'test', or 'all'.
            config (dict): Configuration dictionary containing splitting parameters.
        """
        self.h5_path = h5_path
        self.split = split
        self.config = config or {}
        
        # Extract valid video trial IDs
        self.valid_trial_ids = valid_trials_df['video_trial_id'].dropna().unique()
        
        # The h5 file handle, initialized lazily to avoid multiprocessing issues
        self.h5_file = None
        
        # Build mapping from dataset_idx -> global_h5_idx
        self.frame_indices = self._build_frame_indices(valid_trials_df)

        self.mean_frame = 0.
        if self.config.get('subtract_mean_frame', False):
            mean_frame_path = self.config.get('mean_frame_path')
            if mean_frame_path and os.path.exists(mean_frame_path):
                self.mean_frame = np.load(mean_frame_path)
            else:
                print("Warning: mean_frame_path not provided or file does not exist. Calculating it now...")
                self.mean_frame = self._build_mean_frame(valid_trials_df)

    def _build_frame_indices(self, df):
        """
        Constructs a list of actual HDF5 indices to sample from based on config.
        """
        # Open temporarily just to read the trial IDs
        with h5py.File(self.h5_path, 'r') as h5f:
            h5_trial_ids = h5f['trial_ids'][:]
            
        split_method = self.config.get('frame_selection_method', 'frame_subsample')
        
        if split_method == 'frame_subsample':
            nth_frame = self.config.get('train_nth_frame', 10)
            
            train_indices = []
            test_indices = []
            all_indices = []
            
            # Go trial-by-trial to subsample appropriately
            for trial_id in self.valid_trial_ids:
                trial_mask = (h5_trial_ids == trial_id)
                trial_indices = np.where(trial_mask)[0]
                
                if len(trial_indices) == 0:
                    continue
                    
                # Train gets every nth frame within this trial
                t_train = trial_indices[::nth_frame]
                # Test is offset by half the nth_frame to get a different subset
                t_test = trial_indices[nth_frame//2::nth_frame]
                
                train_indices.append(t_train)
                test_indices.append(t_test)
                all_indices.append(trial_indices)
                
            train_indices = np.concatenate(train_indices) if train_indices else np.array([])
            test_indices = np.concatenate(test_indices) if test_indices else np.array([])
            all_indices = np.concatenate(all_indices) if all_indices else np.array([])
            
            if self.split == 'train':
                return train_indices
            elif self.split == 'test':
                return test_indices
            else: # 'all'
                return np.sort(all_indices)
                
        elif split_method == 'trial_split':
            # Subsample by entire trials
            if self.split == 'train':
                target_trials = df[df['train_split'] == 1]['video_trial_id'].values
            elif self.split == 'test':
                target_trials = df[df['train_split'] == 2]['video_trial_id'].values
            else:
                target_trials = df['video_trial_id'].values
                
            return np.where(np.isin(h5_trial_ids, target_trials))[0]
            
        else:
            raise ValueError(f"Unknown frame_selection_method: {split_method}")

    def _build_mean_frame(self, df):
        """Computes the mean frame across the entire dataset (or a subset) for normalization."""
        with h5py.File(self.h5_path, 'r') as h5f:
            h5_trial_ids = h5f['trial_ids'][:]
            if df is not None:
                indices = np.where(np.isin(h5_trial_ids, df['video_trial_id'].values))[0]
            else:
                indices = np.arange(len(h5_trial_ids))
            n_inds = np.minimum(len(indices), 5000)  # Limit to 5000 frames for efficiency
            subsample_indices = np.linspace(0, len(indices)-1, num=n_inds, dtype=int)  # Sample every 10th frame for efficiency
            mean_frame = np.mean(h5f['frames'][indices[subsample_indices]], axis=0)
        
        return torch.from_numpy(mean_frame).float() / 255.0

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        # 1. LAZY LOADING for multiprocessing safety
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        # 2. Map dataset index to original H5 frame index
        h5_idx = self.frame_indices[idx]
        
        # 3. Read frame from disk
        frame = self.h5_file['frames'][h5_idx]
        
        # 4. Convert and normalize -> [0.0, 1.0]
        # Make sure our bytes become float tensors
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        
        return frame_tensor - self.mean_frame