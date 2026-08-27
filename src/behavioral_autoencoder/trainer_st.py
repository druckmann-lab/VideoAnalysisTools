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
        
        # Modular Scheduler Setup
        if self.config.get('lr_scheduler') == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=100, factor=0.5, min_lr = 1e-6, threshold = 1e-5)
        elif self.config.get('lr_scheduler') == 'CosineAnnealingWarmRestarts':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=500, T_mult=2, eta_min=1e-6)
        else:
            self.scheduler = None
            
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)

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
            
            # Forward pass
            x_recon, z = self.model(batch)
            
            # Loss and backpropagation
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
                
            x_recon, z = self.model(batch)
            loss, recon_loss, latent_loss = self.compute_loss(batch, x_recon, z)
            total_val_loss += loss.item() * batch.size(0)
            total_recon_loss += recon_loss.item() * batch.size(0)
            total_latent_loss += latent_loss.item() * batch.size(0)
            
        return total_val_loss / len(dataloader.dataset), total_recon_loss / len(dataloader.dataset), total_latent_loss / len(dataloader.dataset)

    def fit(self, train_loader, val_loader):
        best_val_loss = float('inf')
        epochs = self.config.get('epochs', 10)
        train_losses = []
        val_losses = []
        val_recon_losses = []
        val_latent_losses = []
        lr_history = []
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)

            ckpt_interval = self.config.get('checkpoint_interval', 500)
            is_ckpt_epoch = epoch % ckpt_interval == (ckpt_interval - 1)

            do_val = (epoch % self.config.get('val_interval', 1) == 0
                      or epoch == 1 or epoch == epochs or is_ckpt_epoch)

            if do_val:
                val_loss, val_recon_loss, val_latent_loss = self.evaluate(val_loader)
            else:
                val_loss = val_recon_loss = val_latent_loss = float('nan')

            #val_loss, val_recon_loss, val_latent_loss = self.evaluate(val_loader)
            
            # Adjust learning rate based on performance
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if do_val:
                        self.scheduler.step(val_loss)
                elif isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                    self.scheduler.step(epoch)
                else:
                    self.scheduler.step()
                    
            current_lr = self.optimizer.param_groups[0]['lr']

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_recon_losses.append(val_recon_loss)
            val_latent_losses.append(val_latent_loss)
            lr_history.append(current_lr)

            # Handle Checkpointing
            if val_loss < best_val_loss and self.config.get('save_best_model', True):
                best_val_loss = val_loss
                print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Recon Loss: {val_recon_loss:.6f} | Val Latent Loss: {val_latent_loss:.6f} | LR: {current_lr:.6e}")
                checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'val_loss': val_loss,
                    'mean_frame_train': train_loader.dataset.mean_frame.cpu().numpy() if hasattr(train_loader.dataset, 'mean_frame') else None,
                    'mean_frame_val': val_loader.dataset.mean_frame.cpu().numpy() if hasattr(val_loader.dataset, 'mean_frame') else None,
                    'loss_mask': self.loss_mask.cpu().numpy() if self.loss_mask is not None else None,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_recon_losses': val_recon_losses,
                    'val_latent_losses': val_latent_losses,
                    'lr_history': lr_history
                }, checkpoint_path)

            if is_ckpt_epoch:
                print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Recon Loss: {val_recon_loss:.6f} | Val Latent Loss: {val_latent_loss:.6f} | LR: {current_lr:.6e}")
                checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'val_loss': val_loss,
                    'mean_frame_train': train_loader.dataset.mean_frame.cpu().numpy() if hasattr(train_loader.dataset, 'mean_frame') else None,
                    'mean_frame_val': val_loader.dataset.mean_frame.cpu().numpy() if hasattr(val_loader.dataset, 'mean_frame') else None,
                    'loss_mask': self.loss_mask.cpu().numpy() if self.loss_mask is not None else None,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_recon_losses': val_recon_losses,
                    'val_latent_losses': val_latent_losses,
                    'lr_history': lr_history
                }, checkpoint_path)