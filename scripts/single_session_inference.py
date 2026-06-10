'''
Inference script for the behavioral video autoencoder.

Loads a trained checkpoint, runs all frames through the model,
and saves two HDF5 files:
  1. <session>_latents.h5           — latent vectors only
  2. <session>_latents_recons.h5    — latent vectors + reconstructed frames

Usage:
    python run_inference.py \
        --checkpoint /path/to/best_model.pt \
        --env aws \
        --animal kd115 \
        --session kd115_twNew_20221206_115814 \
        --output_dir /path/to/output/

2026.06.10. Balint w/ Claude
'''

import os
import sys
import json
import argparse
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader

parent_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir + '/src')

from behavioral_autoencoder.dataset_st import SessionMetadataHandler, H5VideoDataset
from behavioral_autoencoder.models import AutoEncoder


# ─────────────────────────────────────────────
# Config helpers (mirrors train_st.py)
# ─────────────────────────────────────────────

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
        print(f"Warning: Environment config {env_path} not found, using base config only.")

    return config


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run autoencoder inference on all frames")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument('--env', type=str, default='aws', choices=['local', 'aws'],
                        help="Environment config to load (determines paths)")
    parser.add_argument('--animal', type=str, required=True,
                        help="Animal identifier (e.g., kd115)")
    parser.add_argument('--session', type=str, required=True,
                        help="Session identifier (e.g., kd115_twNew_20221206_115814)")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Folder to save output H5 files")
    parser.add_argument('--batch_size', type=int, default=2048,
                        help="Inference batch size (can be larger than training)")
    parser.add_argument('--save_recons', action='store_true', default=True,
                        help="Also save reconstructed frames (disable to save disk space)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Inference Script | Device: {device} ---")

    # ── 1. Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_config = checkpoint['config']['model']
    model = AutoEncoder(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded (trained for {checkpoint.get('epoch', '?')} epochs, "
          f"val_loss={checkpoint.get('val_loss', float('nan')):.6f})")

    # ── 2. Extract mean frame from checkpoint ───────────────────────────────
    # Use the training mean frame for consistency with how the model was trained.
    mean_frame = None
    if checkpoint.get('mean_frame_train') is not None:
        mean_frame = torch.from_numpy(checkpoint['mean_frame_train']).float().to(device)
        print(f"Using training mean frame from checkpoint (shape: {mean_frame.shape})")
    else:
        print("No mean frame found in checkpoint — will not subtract mean.")

    # ── 3. Build dataset (split='all') ───────────────────────────────────────
    config = load_config(args.env)

    metadata_handler = SessionMetadataHandler(
        config=config['metadata_config'],
        mode=args.env,
        animal=args.animal,
        session=args.session
    )
    trial_split_df = metadata_handler.process_all()

    # Use split='all' to cover every valid frame, not just train or test.
    dataset_config = config['dataset'].copy()
    dataset_config['subtract_mean_frame'] = False   # we apply the checkpoint mean manually below

    dataset = H5VideoDataset(
        h5_path=config['dataset']['dataset_path'],
        valid_trials_df=trial_split_df,
        split='all',
        config=dataset_config
    )
    print(f"Dataset ready: {len(dataset)} frames")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,          # must be False — we rely on order to map back to frame_indices
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # ── 4. Run inference ─────────────────────────────────────────────────────
    # We don't know the latent shape until the first batch, so collect into lists first.
    all_latents = []
    all_recons  = []   # only populated if save_recons=True

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # batch shape from dataset: (bs, H, W)  — single-channel grayscale
            batch = batch.to(device)

            # Subtract the checkpoint mean frame if available (matches training behaviour)
            if mean_frame is not None:
                batch = batch - mean_frame

            # Trainer unsqueezes to (bs, 1, H, W) → (bs, 1, 1, H, W) for 5-D model input
            if batch.dim() == 3:
                batch = batch.unsqueeze(1)   # (bs, 1, H, W)
            if batch.dim() == 4:
                batch = batch.unsqueeze(1)   # (bs, 1, 1, H, W)

            x_recon, z = model(batch)
            # z shape:       (bs, seq_len, latent_dim)  or  (bs, latent_dim)
            # x_recon shape: same as input

            all_latents.append(z.cpu().numpy())

            if args.save_recons:
                # Squeeze back to (bs, H, W) for compact storage
                recon = x_recon.squeeze().cpu().numpy()
                if recon.ndim == 2:         # edge case: batch_size=1
                    recon = recon[np.newaxis]
                all_recons.append(recon)

            if (batch_idx + 1) % 20 == 0:
                n_done = min((batch_idx + 1) * args.batch_size, len(dataset))
                print(f"  {n_done}/{len(dataset)} frames processed...")

    print("Inference complete. Assembling outputs...")

    latents_np = np.concatenate(all_latents, axis=0)   # (N, ...) 
    print(f"Latents shape: {latents_np.shape}")

    # ── 5. Build metadata arrays ─────────────────────────────────────────────
    # Map dataset indices back to original H5 trial IDs so the output H5 files
    # carry the same trial_id metadata as the source file.
    frame_indices = dataset.frame_indices.astype(np.int64)      # global h5 indices

    # Read trial_ids from the in-RAM array (already loaded by dataset)
    trial_ids_all = dataset.trial_ids_arr                        # shape: (total_frames,)
    trial_ids_out = trial_ids_all[frame_indices]                 # shape: (N,)

    # ── 6. Save latents-only H5 ──────────────────────────────────────────────
    latents_path = os.path.join(args.output_dir, f"{args.session}_latents.h5")
    print(f"Saving latents to: {latents_path}")
    with h5py.File(latents_path, 'w') as f:
        f.create_dataset('latents',       data=latents_np,   compression='gzip', compression_opts=4)
        f.create_dataset('frame_indices', data=frame_indices, compression='gzip')
        f.create_dataset('trial_ids',     data=trial_ids_out, compression='gzip')
        # Store metadata as attributes for easy inspection
        f.attrs['animal']    = args.animal
        f.attrs['session']   = args.session
        f.attrs['n_frames']  = len(latents_np)
        f.attrs['checkpoint'] = args.checkpoint
        f.attrs['epoch']     = checkpoint.get('epoch', -1)
        f.attrs['val_loss']  = checkpoint.get('val_loss', float('nan'))
    print(f"Saved latents-only H5 ({latents_np.nbytes / 1e6:.1f} MB of latents)")

    # ── 7. Save latents + reconstructions H5 ────────────────────────────────
    if args.save_recons:
        recons_np = np.concatenate(all_recons, axis=0)          # (N, H, W)
        print(f"Reconstructions shape: {recons_np.shape}")

        recons_path = os.path.join(args.output_dir, f"{args.session}_latents_recons.h5")
        print(f"Saving latents + reconstructions to: {recons_path}")
        with h5py.File(recons_path, 'w') as f:
            f.create_dataset('latents',       data=latents_np,   compression='gzip', compression_opts=4)
            f.create_dataset('reconstructions', data=recons_np,  compression='gzip', compression_opts=4)
            f.create_dataset('frame_indices', data=frame_indices, compression='gzip')
            f.create_dataset('trial_ids',     data=trial_ids_out, compression='gzip')
            f.attrs['animal']    = args.animal
            f.attrs['session']   = args.session
            f.attrs['n_frames']  = len(latents_np)
            f.attrs['checkpoint'] = args.checkpoint
            f.attrs['epoch']     = checkpoint.get('epoch', -1)
            f.attrs['val_loss']  = checkpoint.get('val_loss', float('nan'))
        print(f"Saved latents+recons H5 "
              f"({(latents_np.nbytes + recons_np.nbytes) / 1e6:.1f} MB total)")

    print("\n--- Done ---")
    print(f"Output files in: {args.output_dir}")