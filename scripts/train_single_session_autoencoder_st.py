import os
import sys
import gc
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
from behavioral_autoencoder.dataset_st import H5VideoDataset, H5VideoDatasetSequences, build_loss_mask, GpuTensorLoader
from behavioral_autoencoder.models import AutoEncoder

def update_dict(d, u):
    """Recursively updates a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def strip_private(d):
    """
    Drops "_"-prefixed keys recursively.

    JSON has no comments, so env configs carry "_comment" instead; without this
    they would survive the merge into the config.json saved beside every
    checkpoint.
    """
    return {k: (strip_private(v) if isinstance(v, dict) else v)
            for k, v in d.items() if not k.startswith('_')}


def load_config(env_name):
    """
    Loads base config and overwrites with environment specifics.

    An env config may carry "extends": "<other_env>" to inherit from another env
    config before its own keys are applied. This exists so a variant -- e.g. one
    arm of a scheduler A/B -- can be a two-line delta rather than a full copy of
    its parent. Duplicated config files drift, and a drifted arm silently
    invalidates the comparison it was built for.
    """
    with open(f'{parent_dir}/configs/ae_config.json', 'r') as f:
        config = json.load(f)

    chain, name, seen = [], env_name, set()
    while name:
        env_path = f'{parent_dir}/configs/{name}_config.json'
        if not os.path.exists(env_path):
            print(f"Warning: Environment config {env_path} not found.")
            break
        if name in seen:
            raise ValueError(f"circular 'extends' in config chain at {name!r}")
        seen.add(name)
        with open(env_path, 'r') as f:
            env_config = json.load(f)
        # popped so it does not survive into the saved config.json
        name = env_config.pop('extends', None)
        chain.append(strip_private(env_config))

    # Parents first, so a child's keys win over the ones it inherits.
    for env_config in reversed(chain):
        config = update_dict(config, env_config)

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument('--env', type=str, default='local', choices=['local', 'aws','aws_batch'],
                        help="Environment to run in (determines paths and compute specs)")
    parser.add_argument('--animal', type=str, default='kd115', help="Animal identifier (e.g., kd115)")
    parser.add_argument('--session', type=str, default='kd115_twNew_20221206_115814', help="Session identifier (e.g., kd115_twNew_20221206_115814)")
    parser.add_argument('--bpod_path', type=str, default=None, help="Optional path to the Bpod file (overrides config)")
    parser.add_argument('--h5_path', type=str, default=None, help="Optional path to the H5 file (overrides config)")
    parser.add_argument('--mean_frame_path', type=str, default=None, help="Optional path to the mean frame file (overrides config)")

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
    if args.mean_frame_path:
        config['dataset']['mean_frame_path'] = args.mean_frame_path
        
    print(config)

    # Seed before anything that draws: model init and the shuffled epoch order.
    # torch.manual_seed also seeds CUDA, which is what GpuTensorLoader's randperm
    # uses. Without this the two scheduler arms of an A/B would differ by
    # initialisation as well as by schedule.
    seed = config['training'].get('random_seed', 0)
    torch.manual_seed(seed)
    print(f"torch.manual_seed({seed})")

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
        #train_dataset = H5VideoDataset(config['dataset']['dataset_path'], trial_split_df, split='train', config=config['dataset'])
        #val_dataset = H5VideoDataset(config['dataset']['dataset_path'], trial_split_df, split='test', config=config['dataset'])
        # Read the frames ONCE and share them between the two splits.
        frames, trial_ids_arr = H5VideoDataset.load_frames_to_ram(
            config['dataset']['dataset_path'])
        train_dataset = H5VideoDataset(
            config['dataset']['dataset_path'], trial_split_df, split='train',
            config=config['dataset'], frames=frames, trial_ids_arr=trial_ids_arr)
        val_dataset = H5VideoDataset(
            config['dataset']['dataset_path'], trial_split_df, split='test',
            config=config['dataset'], frames=frames, trial_ids_arr=trial_ids_arr)

    elif dataset_type == 'H5VideoDatasetSequences':
        train_dataset = H5VideoDatasetSequences(config['dataset']['dataset_path'], trial_split_df, split='train', config=config['dataset'])
        val_dataset = H5VideoDatasetSequences(config['dataset']['dataset_path'], trial_split_df, split='test', config=config['dataset'])

    # Captured before the host frame tensor is released below.
    frame_shape = train_dataset.frames.shape[1:]

    if dataset_type == 'H5VideoDataset':
        # Both splits move to the GPU as uint8 and are indexed there. See
        # GpuTensorLoader for why this matters more than its 1.12x suggests.
        train_loader = GpuTensorLoader(train_dataset,
                                       batch_size=config['training']['batch_size'],
                                       shuffle=True)
        val_loader = GpuTensorLoader(val_dataset,
                                     batch_size=config['training']['batch_size'],
                                     shuffle=False)

        # The full host tensor holds all ~1.19M frames, but the two splits select
        # only ~166k of them and now have their own GPU copies. Releasing it takes
        # peak RSS from ~19.6 GB to well under 10, which is what lets the run fit
        # a g5.2xlarge. Both datasets alias the same tensor, so both refs must go.
        train_dataset.frames = None
        val_dataset.frames = None
        del frames
        gc.collect()
        print(f"Released host frame tensor; "
              f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val "
              f"frames resident on GPU")
    else:
        # H5VideoDatasetSequences yields sequences via .sequences, not
        # .frame_indices, so GpuTensorLoader does not apply to it.
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
        frame_shape,
        config['dataset'].get('loss_mask_exclude_regions')
    )

    trainer = VideoTrainer(model, config['training'], loss_mask=loss_mask)

    # Begin Training Execution Loop
    trainer.fit(train_loader, val_loader)
