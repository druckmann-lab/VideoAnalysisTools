from torchvision.transforms import v2
import h5py
import re
import os
import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
from tqdm.std import tqdm


def get_video_transforms(config):
    """
    Creates a torchvision v2 transform pipeline based on the config['data'].
    It expects a tensor of shape (C, H, W) or (T, C, H, W) and returns a transformed tensor.
    """
    data_cfg = config['data']
    target_height = data_cfg.get('image_height', 120)
    target_width = data_cfg.get('image_width', 112)
    
    crop_top = data_cfg.get('crop_top', 0)
    crop_bottom = data_cfg.get('crop_bottom', 0)
    crop_left = data_cfg.get('crop_left', 0)
    crop_right = data_cfg.get('crop_right', 0)
    
    transforms_list = []
    
    if crop_top > 0 or crop_bottom > 0 or crop_left > 0 or crop_right > 0:
        transforms_list.append(v2.Lambda(lambda x: x[..., crop_top:(x.shape[-2]-crop_bottom), crop_left:(x.shape[-1]-crop_right)]))
    
    transforms_list.append(v2.Resize(size=(target_height, target_width), antialias=True))

    return v2.Compose(transforms_list)

def convert_videos_to_hdf5(video_folder, h5_filename="session_data.h5", save_path=None, transform=None):
    """
    Sequentially processes all MP4 files in a folder, extracts each frame, applies transforms,
    and saves the frames along with the trial number, filename, and frame index within the trial into an HDF5 file.
    """
    if save_path is None:
        save_path = video_folder
    
    output_file = os.path.join(save_path, h5_filename)
    
    # Gather videos
    video_files = []
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    video_files.sort()
    
    if not video_files:
        print(f"No .mp4 files found in {video_folder}")
        return
    
    print(f"Found {len(video_files)} videos. Starting HDF5 conversion to {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        ds_frames = None
        
        # Initialize metadata datasets
        ds_trial_ids = f.create_dataset("trial_ids", shape=(0,), maxshape=(None,), dtype='i')
        ds_id_within_trial = f.create_dataset("id_within_trial", shape=(0,), maxshape=(None,), dtype='i')
        dt_string = h5py.string_dtype(encoding='utf-8')
        ds_filenames = f.create_dataset("filenames", shape=(0,), maxshape=(None,), dtype=dt_string)
        
        current_frame_idx = 0
        
        for vid_path in tqdm(video_files, desc="Processing Videos"):
            # Extract trial number using regex e.g., "_trial111_" -> 111
            match = re.search(r'_trial(\d+)_', os.path.basename(vid_path))
            trial_id = int(match.group(1)) if match else -1
            filename_str = os.path.basename(vid_path)
            
            decoder = VideoDecoder(vid_path)
            num_frames = len(decoder)
            
            if num_frames == 0:
                continue
                
            # --- OPTIMIZATION 1: Resize the dataset ONCE per video ---
            new_total_frames = current_frame_idx + num_frames
            
            # If this is the first video, peek at a frame to initialize the dataset shape
            if ds_frames is None:
                sample_frame = decoder.get_frame_at(0).data[0:1, :, :]
                if transform:
                    sample_frame = transform(sample_frame)
                ch, h, w = sample_frame.shape
                
                # OPTIMIZATION 2: Enforce uint8 and specify disk chunks
                ds_frames = f.create_dataset(
                    "frames", 
                    shape=(0, ch, h, w), 
                    maxshape=(None, ch, h, w), 
                    dtype='uint8',           # Enforce 1-byte per pixel
                    chunks=(256, ch, h, w),  # Chunk size optimized for read speed
                    compression="lzf"
                )

            # Resize datasets to hold the entire current video
            ds_frames.resize(new_total_frames, axis=0)
            ds_trial_ids.resize(new_total_frames, axis=0)
            ds_id_within_trial.resize(new_total_frames, axis=0)
            ds_filenames.resize(new_total_frames, axis=0)
            
            # --- OPTIMIZATION 3: Process frames and write into the pre-allocated slice ---
            for i in range(num_frames):
                frame_tensor = decoder.get_frame_at(i).data[0:1, :, :]
                
                if transform:
                    frame_tensor = transform(frame_tensor)
                
                # Safely handle float-to-uint8 conversion if your transform scales to [0.0, 1.0]
                if frame_tensor.is_floating_point():
                    if frame_tensor.max() <= 1.0:
                        frame_tensor = (frame_tensor * 255.0).clamp(0, 255)
                    frame_tensor = frame_tensor.to(torch.uint8)
                else:
                    frame_tensor = frame_tensor.to(torch.uint8)
                
                numpy_frame = frame_tensor.numpy()
                
                # Write directly to the pre-allocated index
                write_idx = current_frame_idx + i
                ds_frames[write_idx] = numpy_frame
                ds_trial_ids[write_idx] = trial_id
                ds_id_within_trial[write_idx] = i
                ds_filenames[write_idx] = filename_str
                
            current_frame_idx = new_total_frames
            
    print(f"Done! Successfully packed {current_frame_idx} frames into {output_file}")
