import os
import sys
import json
import torch
import argparse
from torch.utils.data import DataLoader
import datetime

parent_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Parent directory: {parent_dir}")
sys.path.append(parent_dir +'/src')

from behavioral_autoencoder.dataset_st import SessionMetadataHandler
from behavioral_autoencoder.trainer_st import VideoTrainer
from behavioral_autoencoder.dataset_st import H5VideoDataset, H5VideoDatasetSequences, build_loss_mask
from behavioral_autoencoder.models import AutoEncoder

def update_dict(d, u):
    """Recursively updates a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_config(env_name):
    """Loads base config and overwrites with environment specifics."""
    with open(f'{parent_dir}/configs/ae_config.json', 'r') as f:
        config = json.load(f)
        
    env_path = f'{parent_dir}/configs/{env_name}_config.json'
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            env_config = json.load(f)
        config = update_dict(config, env_config)
    else:
        print(f"Warning: Environment config {env_path} not found.")
        
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument('--env', type=str, default='local', choices=['local', 'aws','aws_batch'],
                        help="Environment to run in (determines paths and compute specs)")
    parser.add_argument('--animal', type=str, default='kd115', help="Animal identifier (e.g., kd115)")
    parser.add_argument('--session', type=str, default='kd115_twNew_20221206_115814', help="Session identifier (e.g., kd115_twNew_20221206_115814)")
    parser.add_argument('--bpod_path', type=str, default=None, help="Optional path to the Bpod file (overrides config)")
    parser.add_argument('--h5_path', type=str, default=None, help="Optional path to the H5 file (overrides config)")

    args = parser.parse_args()

    print(f"--- Initializing Training Pipeline ({args.env.upper()} Environment) ---")
    
    # 1. Load merged configurations
    config = load_config(args.env)
    animal = args.animal
    session = args.session
    # overwrite paths if provided
    if args.bpod_path:
        config['metadata_config']['bpod_path'] = args.bpod_path
    if args.h5_path:
        config['metadata_config']['h5_path'] = args.h5_path
        config['dataset']['dataset_path'] = args.h5_path
        
    print(config)

    # Initialize Dataset and Loaders

    metadata_handler = SessionMetadataHandler(
    config=config['metadata_config'], 
    mode='local', 
    animal=animal, 
    session=session
    )

    trial_split_df = metadata_handler.process_all()

    dataset_type = config['dataset'].get('type', 'H5VideoDataset')
    if dataset_type == 'H5VideoDataset':
        train_dataset = H5VideoDataset(config['dataset']['dataset_path'], trial_split_df, split='train', config=config['dataset'])
        val_dataset = H5VideoDataset(config['dataset']['dataset_path'], trial_split_df, split='test', config=config['dataset'])
    elif dataset_type == 'H5VideoDatasetSequences':
        train_dataset = H5VideoDatasetSequences(config['dataset']['dataset_path'], trial_split_df, split='train', config=config['dataset'])
        val_dataset = H5VideoDatasetSequences(config['dataset']['dataset_path'], trial_split_df, split='test', config=config['dataset'])

    train_loader = DataLoader(train_dataset, 
                              batch_size=config['training']['batch_size'], 
                              shuffle=True, 
                              num_workers=8, 
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=False, 
                            num_workers=8, 
                            pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True)

    # Initialize Model and Run Trainer
    model = AutoEncoder(config['model'])

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = config['training']['checkpoint_dir'] + f"{animal}/{session}_{date_str}/"
    os.makedirs(save_folder, exist_ok=True)
    config['training']['checkpoint_dir'] = save_folder

    # save the configs at the end of training
    with open(save_folder + "config.json", "w") as f:
        json.dump(config, f)

    # Optionally exclude hand-picked distractor regions (e.g. lickspout) from the recon loss
    loss_mask = build_loss_mask(
        train_dataset.frames.shape[1:],
        config['dataset'].get('loss_mask_exclude_regions')
    )

    trainer = VideoTrainer(model, config['training'], loss_mask=loss_mask)

    # Begin Training Execution Loop
    trainer.fit(train_loader, val_loader)
