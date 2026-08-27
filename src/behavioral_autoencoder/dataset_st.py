'''
Dataset and metadata handling for Tim's dataset for the Spatial Transcriptomics project.


2026.06.08. Balint
'''

import os
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split


def build_loss_mask(shape, exclude_regions=None):
    """
    Builds a (H, W) float mask of ones with zeros inside hand-picked rectangles.

    Used to exclude session-specific distractor regions (e.g. a lickspout) from the
    reconstruction loss so the encoder isn't pressured to spend latent capacity on them.

    Args:
        shape (tuple): (H, W) of the frames as loaded from the h5 file (post-crop space).
        exclude_regions (list[dict], optional): each dict has 'top', 'bottom', 'left', 'right'
            pixel bounds (top/left inclusive, bottom/right exclusive) of a region to exclude.

    Returns:
        torch.FloatTensor of shape (H, W), or None if no regions are given.
    """
    if not exclude_regions:
        return None

    mask = np.ones(shape, dtype=np.float32)
    for region in exclude_regions:
        mask[region['top']:region['bottom'], region['left']:region['right']] = 0.0

    return torch.from_numpy(mask)


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
        if 'bpod_path' in self.config:
            self.bpod_path = self.config['bpod_path']
        else:
            self.bpod_path = f"{self.config['bpod_folder']}{self.animal}/{self.session}.bpod.npy"
        if 'h5_path' in self.config:
            self.h5_path = self.config['h5_path']
        elif 'h5_filename' in self.config:
            self.h5_path = f"{self.config['h5_folder']}{self.session}/{self.config['h5_filename']}"
        else:
            self.h5_path = f"{self.config['h5_folder']}{self.session}/{self.session}_side.h5"

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

        print("Loading dataset into RAM...")
        with h5py.File(self.h5_path, 'r') as f:
            self.frames = torch.from_numpy(
                f['frames'][:])   # shape: (N, H, W), lives in RAM
            self.trial_ids_arr = f['trial_ids'][:]
        print(f"Loaded {len(self.frames)} frames into RAM")
        
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
                mean_frame_np = np.load(mean_frame_path)
                self.mean_frame = torch.from_numpy(mean_frame_np)
            else:
                print("Warning: mean_frame_path not provided or file does not exist. Calculating it now...")
                self.mean_frame = self._build_mean_frame(valid_trials_df)

    def _build_frame_indices(self, df):
        """
        Constructs a list of actual HDF5 indices to sample from based on config.
        """
        # Open temporarily just to read the trial IDs
        #with h5py.File(self.h5_path, 'r') as h5f:
        #    h5_trial_ids = h5f['trial_ids'][:]
        h5_trial_ids = self.trial_ids_arr

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
            n_inds = np.minimum(len(indices), 50000)  # Calc with 50k frames
            subsample_indices = np.linspace(0, len(indices)-1, num=n_inds, dtype=int)
            mean_frame = np.mean(h5f['frames'][indices[subsample_indices]], axis=0)
        
        return torch.from_numpy(mean_frame).float() / 255.0

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        h5_idx = self.frame_indices[idx]
        return self.frames[h5_idx].float() / 255.0 - self.mean_frame
    

class H5VideoDatasetSequences(Dataset):
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

        print("Loading dataset into RAM...")
        with h5py.File(self.h5_path, 'r') as f:
            self.frames = torch.from_numpy(
                f['frames'][:])   # shape: (N, H, W), lives in RAM
            self.trial_ids_arr = f['trial_ids'][:]
        print(f"Loaded {len(self.frames)} frames into RAM")
        
        # Extract valid video trial IDs
        self.valid_trial_ids = valid_trials_df['video_trial_id'].dropna().unique()
        
        # Build mapping from dataset_idx -> global_h5_idx
        self.sequences = self._build_sequences(valid_trials_df)

        self.mean_frame = 0.
        if self.config.get('subtract_mean_frame', False):
            mean_frame_path = self.config.get('mean_frame_path')
            if mean_frame_path and os.path.exists(mean_frame_path):
                self.mean_frame = np.load(mean_frame_path)
            else:
                print("Warning: mean_frame_path not provided or file does not exist. Calculating it now...")
                self.mean_frame = self._build_mean_frame(valid_trials_df)

    def _build_sequences(self, df):
        """
        Constructs a list of actual HDF5 indices to sample from based on config.
        """
        # Open temporarily just to read the trial IDs
        #with h5py.File(self.h5_path, 'r') as h5f:
        #    h5_trial_ids = h5f['trial_ids'][:]
        h5_trial_ids = self.trial_ids_arr

        split_method = self.config.get('frame_selection_method', 'frame_subsample')
        sequence_num_frames = self.config.get('sequence_num_frames', 1280)
        
        if split_method == 'frame_subsample':
            nth_frame = self.config.get('train_nth_frame', 10)
            
            train_indices = []
            test_indices = []
            all_indices = []
            
            # Go trial-by-trial to subsample appropriately
            for trial_id in self.valid_trial_ids:
                trial_mask = (h5_trial_ids == trial_id)
                trial_indices = np.where(trial_mask)[0]
                
                if len(trial_indices) == 0 or len(trial_indices) < sequence_num_frames:
                    continue
                    
                # Train gets every nth frame within this trial
                t_train = trial_indices[0:sequence_num_frames:nth_frame]
                # Test is offset by half the nth_frame to get a different subset
                t_test = trial_indices[nth_frame//2:(sequence_num_frames+nth_frame//2):nth_frame]
                
                train_indices.append(t_train)
                test_indices.append(t_test)
                
                t_all = trial_indices[0:sequence_num_frames:nth_frame] 
                all_indices.append(t_all)
                
            train_sequences = np.array(train_indices) if train_indices else np.array([])
            test_sequences = np.array(test_indices) if test_indices else np.array([])
            all_sequences = np.array(all_indices) if all_indices else np.array([])
            
            if self.split == 'train':
                return train_sequences
            elif self.split == 'test':
                return test_sequences
            else: # 'all'
                return all_sequences
                
        elif split_method == 'trial_split':
            # Subsample by entire trials
            if self.split == 'train':
                target_trials = df[df['train_split'] == 1]['video_trial_id'].values
            elif self.split == 'test':
                target_trials = df[df['train_split'] == 2]['video_trial_id'].values
            else:
                target_trials = df['video_trial_id'].values
            sequences = []
            for trial_id in np.where(np.isin(h5_trial_ids, target_trials))[0]:
                trial_mask = (h5_trial_ids == trial_id)
                trial_indices = np.where(trial_mask)[0]
                if len(trial_indices) == 0 or len(trial_indices) < sequence_num_frames:
                    continue
                t_trial = trial_indices[0:sequence_num_frames:nth_frame]
                sequences.append(t_trial)

            return np.array(sequences)
            
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
            n_inds = np.minimum(len(indices), 50000)  # Calc with 50k frames
            subsample_indices = np.linspace(0, len(indices)-1, num=n_inds, dtype=int)
            mean_frame = np.mean(h5f['frames'][indices[subsample_indices]], axis=0)
        
        return torch.from_numpy(mean_frame).float() / 255.0

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        h5_inds = self.sequences[idx]
        return self.frames[h5_inds].float() / 255.0 - self.mean_frame