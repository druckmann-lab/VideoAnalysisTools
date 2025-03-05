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
    seq_length : int    
        we end up outputting data which has a sequence dimension for consistency with other implementations as a singleton dimension preceding the others. 
    """

    def __init__(self, base_folder, extension=".png", trial_pattern=None, transform = None, seq_length = 1):
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
        self.seq_length = seq_length
        assert seq_length == 1, "can't do more than this. "
        
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
        return img[None,:]


