"""
Test functions within the dataset class.
Uses test data inside `test_data` folder. 
"""
from behavioral_autoencoder.dataset import SessionFramesDataset
import pytest
import numpy as np
import os
from pathlib import Path
import cv2
import tarfile
import shutil

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
    example_images = np.load('./test_data/example_images.npy')
    
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

@pytest.fixture
def temp_hierarchical_folder(tmp_path):
    return temp_hierarchical_folder_generator(
        tmp_path, 
        n_trials=3, 
        n_ims_per_trial=10, 
        extra_files=None
    )

@pytest.fixture
def temp_hierarchical_folder_extra(tmp_path):
    """Same as temp_hierarchical_folder but with extra files"""
    return temp_hierarchical_folder_generator(
        tmp_path, 
        n_trials=3, 
        n_ims_per_trial=10, 
        extra_files=['metadata.txt', 'notes.txt', ".DS_Store"]
    )

@pytest.fixture
def temp_hierarchical_archive(tmp_path, n_trials=3, n_ims_per_trial=10, extra_files=None):
    """Creates a tar archive containing a hierarchical folder structure for testing.
    
    This fixture first creates the same folder structure as temp_hierarchical_folder,
    then compresses it into a tar archive.
    
    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing temporary directory path
    temp_hierarchical_folder : Path
        Pytest fixture providing the hierarchical folder structure
    n_trials : int, optional
        Number of trial folders to create, by default 3
    n_ims_per_trial : int, optional
        Number of images to create per trial, by default 10
    extra_files : list of str, optional
        List of extra file names to create in each trial directory
        
    Returns
    -------
    Path
        Path to the created tar archive
    """
    # Create the archive
    archive_path = tmp_path / "temp_session.tar.gz"
    
    with tarfile.open(archive_path, "w:gz") as tar:
        # Add the entire session directory to the archive
        tar.add(temp_hierarchical_folder_generator(tmp_path,n_trials=n_trials,n_ims_per_trial=n_ims_per_trial,extra_files=extra_files), arcname=temp_hierarchical_folder.name)
    
    return archive_path

def test_transform_image():
    """
    If I understand transform_image correctly, it is defined in terms of the target output pixel shape. That is, we crop the top `np.ceil(crop_info["h_coord"]/target_shape[0])`% of the image, and then reshape to `target_shape`. We can decorrelate these parameters. 
    """

class Test_SessionFramesDataset:
    def test_init(self, temp_hierarchical_folder):
        """Test the SessionFramesDataset initialization and basic properties.
        
        Tests:
        1. Dataset can be initialized
        2. Number of trials matches fixture
        3. Image dimensions and format are correct
        4. Dataset length matches expected total frames
        """
        dataset = SessionFramesDataset(temp_hierarchical_folder)
        
        # Test number of trials
        assert len(dataset.trial_folders) == 3, "Should have 3 trials by default"
        
        # Test image properties
        first_image = dataset[0]
        assert isinstance(first_image, np.ndarray), "Dataset should return numpy arrays"
        assert len(first_image.shape) == 2, "Images should be 2D grayscale (H,W)"
        assert first_image.dtype == np.float32, "Images should be float32"
        
        # Test dataset length
        expected_length = 3 * 10  # n_trials * n_ims_per_trial
        assert len(dataset) == expected_length, f"Dataset should have {expected_length} total frames"
        
        # Test all images are readable
        for i in range(len(dataset)):
            img = dataset[i]
            assert img is not None, f"Failed to load image at index {i}"

    def test_extra_frame_files(self, temp_hierarchical_folder_extra):
        """
        test for other files which are included in frame directories. 
        """
        dataset = SessionFramesDataset(temp_hierarchical_folder_extra)
        trialdirs = os.listdir(temp_hierarchical_folder_extra)
        print(os.listdir(os.path.join(temp_hierarchical_folder_extra, trialdirs[0])))
        assert len(dataset) == 30  # 3 trials * 10 images per trial
        len(dataset)

    def test_extra_dir_files(self, temp_hierarchical_folder_extra, tmp_path):
        """
        Test that the dataset properly filters trial directories when there are extra files
        and folders in the base directory.
        """
        # Add some extra files and folders to the base directory
        base_dir = temp_hierarchical_folder_extra
        
        # Create extra files
        (base_dir / "metadata.json").touch()
        (base_dir / ".DS_Store").touch()
        
        # Create a non-trial directory
        other_dir = base_dir / "other_data"
        other_dir.mkdir()
        (other_dir / "some_file.txt").touch()
        
        # Test with no pattern (should include all directories including 'other_data')
        dataset = SessionFramesDataset(base_dir)
        assert len(dataset.trial_folders) == 4  # 3 trial folders + other_data
        assert "other_data" in dataset.trial_folders
        
        # Test with specific pattern for trial folders
        dataset_with_pattern = SessionFramesDataset(base_dir, trial_pattern=r"^\d+_trial$")
        assert len(dataset_with_pattern.trial_folders) == 3
        assert all(folder.endswith("_trial") for folder in dataset_with_pattern.trial_folders)
        assert "other_data" not in dataset_with_pattern.trial_folders
        
        # Test with pattern that should match nothing
        dataset_no_matches = SessionFramesDataset(base_dir, trial_pattern=r"^nonexistent.*$")
        assert len(dataset_no_matches.trial_folders) == 0

    def test_getitem_performance(self, tmp_path):
        """
        Test performance difference between searchsorted and argmax methods in __getitem__
        using a larger dataset.
        """
        import time
        
        # Create a larger dataset
        n_trials = 100  # Increased number of trials
        n_ims_per_trial = 50  # Increased images per trial
        session_dir = temp_hierarchical_folder_generator(
            tmp_path, 
            n_trials=n_trials,
            n_ims_per_trial=n_ims_per_trial
        )
        
        dataset = SessionFramesDataset(session_dir)
        n_items = len(dataset)
        n_iterations = 1000  # Number of random accesses to test
        
        # Generate random indices to access
        np.random.seed(42)  # For reproducibility
        random_indices = np.random.randint(0, n_items, n_iterations)
        
        # Test argmax method
        start_time = time.time()
        for idx in random_indices:
            _ = dataset.__getitem__(idx, method="argmax")
        argmax_time = time.time() - start_time
        
        # Test searchsorted method
        start_time = time.time()
        for idx in random_indices:
            _ = dataset.__getitem__(idx, method="searchsorted")
        searchsorted_time = time.time() - start_time
        
        # Print results for inspection
        print(f"\nPerformance comparison over {n_iterations} accesses:")
        print(f"Dataset size: {n_trials} trials, {n_ims_per_trial} images per trial")
        print(f"argmax method: {argmax_time:.4f} seconds")
        print(f"searchsorted method: {searchsorted_time:.4f} seconds")
        print(f"Speed improvement: {(argmax_time/searchsorted_time):.2f}x faster")
        
        # Assert searchsorted is faster
        assert searchsorted_time < argmax_time, \
            f"searchsorted ({searchsorted_time:.4f}s) should be faster than argmax ({argmax_time:.4f}s)"
        
        # Verify both methods return the same results
        for idx in random_indices[:10]:  # Check first 10 indices
            result_argmax = dataset.__getitem__(idx, method="argmax")
            result_searchsorted = dataset.__getitem__(idx, method="searchsorted")
            assert np.array_equal(result_argmax, result_searchsorted), \
                f"Methods returned different results for index {idx}"


