"""
Utils to take an input video and write individual frames to a directory. 
"""
import cv2
import fire
import os
from pathlib import Path
import logging

def extract_frames(video_path: str, output_dir: Path, frame_prefix: str = "frame") -> int:
    """Extract frames from a video file and save them as PNG images.
    
    Parameters
    ----------
    video_path: str
        to the video file
    output_dir: str
        Directory to save the extracted frames
    frame_prefix : str
        Prefix for frame filenames
    
    Returns:
        int: Number of frames extracted
    
    Raises:
        ValueError: If video file cannot be opened
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    
    frame_paths = {} 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame as PNG
        frame_path = output_dir / f"{frame_prefix}_{frame_count:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths[frame_count] = frame_path
        frame_count += 1
    
    cap.release()
    return frame_paths

def check_exists(frame_dir: Path) -> bool:
    """Check if the frame directory exists. If it does, ask user if they want to proceed.
    Parameters
    ----------
    frame_dir : Path
       path to frames
          path to frames
    """
    
    if frame_dir.exists():
        response = input(f"Directory {frame_dir} already exists. Do you want to proceed and overwrite? (y/n)")
        return response == "y"
    return True

def main(video_dir,frame_dir,video_suffix=".avi"):
    """Given a directory of videos, write to a different directory with the following structure:
        1. one subdirectory per video file, as well as a metadata file `annotations.txt`.
        2. within each subdirectory, pngs per individual frames.

        Will ask if we want to proceed if the folder already exists. 
        Will throw a warning if we have issues pulling out frames. 
        Parameters 
        ----------
            video_dir: directory containing video files. 
            frame_dir: directory to write frames to. 
            video_suffix (default=".avi"): suffix of video files to consider. 
    """
    video_dir = Path(video_dir)
    frame_dir = Path(frame_dir)
    # 1. Get a directory which contains video files. Store video file names.  
    video_files = os.listdir(video_dir)
    video_files = [f for f in video_files if f.endswith(video_suffix)]
    # 2. Check that the directory we care about exists. If it doesn't create. If it does, ask user.  
    video_files_write = []
    for video_file in video_files:
        if check_exists(frame_dir / video_file):
            video_files_write.append(video_file)

    # 3. For each video, extract all frames into the directory. 
    for video_file in video_files_write:
        print(f"Processing {video_file}")
        video_dir_name = video_file.split(".")[0]
        frame_paths = extract_frames(os.path.join(video_dir, video_file), frame_dir / video_dir_name)
        # 4. Write the frame paths to a metadata file.
        first,last = list(frame_paths.keys())[0],list(frame_paths.keys())[-1]
        # Write to annotations file immediately and close it
        with open(frame_dir / "annotations.txt", "a") as f:
            f.write(f"{video_dir_name} {first} {last} 0\n")

def npy_to_frames()

if __name__ == "__main__":
    fire.Fire(main)
