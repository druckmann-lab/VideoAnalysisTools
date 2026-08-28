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
# One definition, shared with single_session_inference.py and the benchmark:
# training and inference must resolve a checkpoint's env identically.
from behavioral_autoencoder.config import load_config, update_dict, strip_private

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    # Deliberately not a `choices` list: env names are open-ended (variants like
    # aws_batch_fastcycle are just another configs/<env>_config.json), and a
    # stale choices list would reject a valid config with argparse exit 2 --
    # after the instance has already staged 14 GB. load_config validates instead.
    parser.add_argument('--env', type=str, default='local',
                        help="Environment to run in: configs/<env>_config.json")
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
