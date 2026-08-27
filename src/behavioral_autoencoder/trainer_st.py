'''
Trainer class for behavioral autoencoder training on Tim's Spatial Transcriptomics dataset.

2026.06.08. Balint
'''

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import torch.nn as nn
import os
from contextlib import nullcontext

# bf16 needs no GradScaler on Ampere and later; fp16 would, so it is offered here
# for completeness but is not what the sweep uses.
_AMP_DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16}


class VideoTrainer:
    def __init__(self, model, config, device=None, loss_mask=None):
        self.model = model
        self.config = config

        # Setup Compute Device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Core Optimization Components
        self.optimizer = AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        self.mse_criterion = nn.MSELoss()
        self.latent_lambda = self.config.get('latent_lambda', 0.0)

        # Optional (H, W) mask excluding distractor regions (e.g. lickspout) from recon loss.
        # Broadcasts against the (bs, seq, 1, H, W) frame tensors.
        self.loss_mask = loss_mask.to(self.device) if loss_mask is not None else None

        # Mixed precision. bf16 is the single largest speedup measured on an A10G
        # (7.72s -> 5.60s per epoch, 1.38x); it wins mostly by halving memory
        # traffic, since 16-channel convs cannot fill tensor-core tiles anyway.
        amp_dtype = self.config.get('amp_dtype')
        if amp_dtype is not None and amp_dtype not in _AMP_DTYPES:
            raise ValueError(
                f"amp_dtype must be one of {sorted(_AMP_DTYPES)} or null, "
                f"got {amp_dtype!r}")
        self.amp_dtype = _AMP_DTYPES.get(amp_dtype)

        # Modular Scheduler Setup
        if self.config.get('lr_scheduler') == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=100, factor=0.5, min_lr = 1e-6, threshold = 1e-5)
        elif self.config.get('lr_scheduler') == 'CosineAnnealingWarmRestarts':
            # Cycle boundaries are T_0*(T_mult**n - 1)/(T_mult - 1) for T_mult>1,
            # or multiples of T_0 for T_mult==1. Stopping training on a boundary
            # is what leaves the final epoch fully annealed at eta_min.
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('T_0', 500),
                T_mult=self.config.get('T_mult', 2),
                eta_min=self.config.get('eta_min', 1e-6))
        else:
            self.scheduler = None

        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)

    def _autocast(self):
        if self.amp_dtype is None or self.device.type != 'cuda':
            return nullcontext()
        return torch.autocast('cuda', dtype=self.amp_dtype)

    def compute_loss(self, x, x_recon, z):
        """
        Calculates the joint loss function: Total = MSE + lambda * L2_norm(z)
        """
        # Reconstruction loss
        if self.loss_mask is not None:
            sq_err = (x_recon - x) ** 2
            # Normalize by the number of unmasked elements only, so the loss magnitude
            # (and the recon/latent balance set by latent_lambda) stays comparable to
            # the unmasked case.
            n_valid = sq_err.numel() / self.loss_mask.numel() * self.loss_mask.sum()
            recon_loss = (sq_err * self.loss_mask).sum() / n_valid
        else:
            recon_loss = self.mse_criterion(x_recon, x)

        # L2 Regularization on the latent space
        # z shape is (bs, seq_length, embed_size). Normalizing by element count.
        latent_loss = torch.mean(torch.sum(z ** 2, dim=-1))
        
        total_loss = recon_loss + (self.latent_lambda * latent_loss)
        
        return total_loss, recon_loss, latent_loss

    def train_epoch(self, dataloader):
        self.model.train()
        total_epoch_loss = 0.0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            
            # Adapt 4D DataLoader batch [bs, c, h, w] to 5D Model expectation [bs, 1, c, h, w]
            if batch.dim() == 4:
                batch = batch.unsqueeze(1)
                
            self.optimizer.zero_grad()

            # Forward pass and loss under autocast; backward stays outside it,
            # per the torch.amp recipe.
            with self._autocast():
                x_recon, z = self.model(batch)
                loss, _, _ = self.compute_loss(batch, x_recon, z)

            loss.backward()
            self.optimizer.step()
            
            total_epoch_loss += loss.item() * batch.size(0)
            
        return total_epoch_loss / len(dataloader.dataset)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_val_loss = 0.0
        total_recon_loss = 0.0
        total_latent_loss = 0.0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            if batch.dim() == 4:
                batch = batch.unsqueeze(1)
                
            with self._autocast():
                x_recon, z = self.model(batch)
                loss, recon_loss, latent_loss = self.compute_loss(batch, x_recon, z)

            total_val_loss += loss.item() * batch.size(0)
            total_recon_loss += recon_loss.item() * batch.size(0)
            total_latent_loss += latent_loss.item() * batch.size(0)
            
        return total_val_loss / len(dataloader.dataset), total_recon_loss / len(dataloader.dataset), total_latent_loss / len(dataloader.dataset)

    def _mean_frame_of(self, loader):
        """mean_frame is a tensor when subtract_mean_frame is on, else the float 0."""
        mf = getattr(loader.dataset, 'mean_frame', None)
        return mf.cpu().numpy() if torch.is_tensor(mf) else mf

    def _save_checkpoint(self, path, epoch, train_loader, val_loader, val_loss,
                         histories):
        """
        The single definition of the checkpoint payload.

        There are three call sites -- best, periodic and final -- and keeping
        three copies of this dict in step by hand is how they drift apart.
        `histories` holds references to the live lists, so it always reflects
        everything appended so far.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'val_loss': val_loss,
            'mean_frame_train': self._mean_frame_of(train_loader),
            'mean_frame_val': self._mean_frame_of(val_loader),
            'loss_mask': self.loss_mask.cpu().numpy() if self.loss_mask is not None else None,
            **histories,
        }, path)

    def fit(self, train_loader, val_loader):
        best_val_loss = float('inf')
        epochs = self.config.get('epochs', 10)
        train_losses = []
        val_losses = []
        val_recon_losses = []
        val_latent_losses = []
        lr_history = []

        # Values, not copies: the lists are mutated in place, so every save picks
        # up the full history without rebuilding this dict.
        histories = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_recon_losses': val_recon_losses,
            'val_latent_losses': val_latent_losses,
            'lr_history': lr_history,
        }

        # Defined up front so a zero-epoch config still saves something coherent.
        val_loss = val_recon_loss = val_latent_loss = float('nan')

        for epoch in range(1, epochs + 1):
            # The LR this epoch actually trains with. The scheduler is stepped
            # AFTER train_epoch, so the post-step value belongs to epoch+1;
            # recording that shifts lr_history by one and reports the peak LR on
            # the final epoch, exactly where the cosine has just restarted.
            lr_used = self.optimizer.param_groups[0]['lr']

            train_loss = self.train_epoch(train_loader)

            ckpt_interval = self.config.get('checkpoint_interval', 500)
            is_ckpt_epoch = epoch % ckpt_interval == (ckpt_interval - 1)

            do_val = (epoch % self.config.get('val_interval', 1) == 0
                      or epoch == 1 or epoch == epochs or is_ckpt_epoch)

            if do_val:
                val_loss, val_recon_loss, val_latent_loss = self.evaluate(val_loader)
            else:
                val_loss = val_recon_loss = val_latent_loss = float('nan')

            # Adjust the learning rate for the NEXT epoch.
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if do_val:
                        self.scheduler.step(val_loss)
                elif isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                    self.scheduler.step(epoch)
                else:
                    self.scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_recon_losses.append(val_recon_loss)
            val_latent_losses.append(val_latent_loss)
            lr_history.append(lr_used)

            status = (f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} "
                      f"| Val Loss: {val_loss:.6f} | Val Recon Loss: {val_recon_loss:.6f} "
                      f"| Val Latent Loss: {val_latent_loss:.6f} | LR: {lr_used:.6e}")

            # Handle Checkpointing
            if val_loss < best_val_loss and self.config.get('save_best_model', True):
                best_val_loss = val_loss
                print(status)
                self._save_checkpoint(
                    os.path.join(self.config['checkpoint_dir'], 'best_model.pt'),
                    epoch, train_loader, val_loader, val_loss, histories)

            if is_ckpt_epoch:
                print(status)
                self._save_checkpoint(
                    os.path.join(self.config['checkpoint_dir'],
                                 f'checkpoint_epoch_{epoch}.pt'),
                    epoch, train_loader, val_loader, val_loss, histories)

        # Unconditional final save. is_ckpt_epoch fires on epoch % N == N-1, so
        # the last epoch never gets a periodic checkpoint, and best_model.pt only
        # holds it if it happened to improve val loss. After an 11h run that is
        # not a good enough guarantee.
        print(f"Training finished at epoch {epochs}; writing final_model.pt")
        self._save_checkpoint(
            os.path.join(self.config['checkpoint_dir'], 'final_model.pt'),
            epochs, train_loader, val_loader, val_loss, histories)
