import pytorch_lightning as pl
import torch
import torch.nn as nn
from behavioral_autoencoder.metrics import L2_loss
from behavioral_autoencoder.networks import SingleSessionAutoEncoder
from torch.optim.lr_scheduler import LinearLR,ChainedScheduler

models = {
        "single_session_autoencoder":SingleSessionAutoEncoder
        }

class Autoencoder_Models(pl.LightningModule):
    """
    Abstract base class for Autoencoder models. Handles things like loss definition, model choice,  
    """
    def __init__(self,hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.reconstruct_criterion = nn.MSELoss()
        self.shrink_criterion = L2_loss
    def forward(self,batch):
        images = batch
        latents,predictions = self.model(images)
        return latents,predictions
    def training_step(self, batch, batch_nb):
        predictions,latents = self.forward(batch)
        rec_loss = self.reconstruct_criterion(predictions,batch)
        shrink_loss = self.shrink_criterion(latents)
        loss = rec_loss+self.hparams["train_config"]["l2_weight"]*shrink_loss
        self.log("loss/train", loss)
        self.log("mse/train", rec_loss)
        return loss
    def validation_step(self, batch, batch_nb):
        predictions,latents = self.forward(batch)
        rec_loss = self.reconstruct_criterion(predictions,batch)
        shrink_loss = self.shrink_criterion(latents)
        loss = rec_loss+self.hparams["train_config"]["l2_weight"]*shrink_loss
        self.log("loss/val", loss)
        self.log("mse/val", rec_loss)
    def test_step(self, batch, batch_nb):
        predictions,latents = self.forward(batch)
        rec_loss = self.reconstruct_criterion(predictions,batch)
        shrink_loss = self.shrink_criterion(latents)
        loss = rec_loss+self.hparams["train_config"]["l2_weight"]*shrink_loss
        self.log("loss/val", loss)
        self.log("mse/val", rec_loss)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["train_config"]["learning_rate"],
            weight_decay=self.hparams["train_config"]["weight_decay"],
        )
        scheduler = self.setup_scheduler(optimizer)
        return [optimizer], [scheduler]
    def setup_scheduler(self,optimizer):
        """Chooses between the cosine learning rate scheduler, linear scheduler, or step scheduler. 

        """
        if self.hparams["train_config"]["scheduler"] == "linear":
            scheduler = {
                    "scheduler": LinearLR(
                        optimizer,start_factor=self.hparams["train_config"]["start_factor"],
                        end_factor=self.hparams["train_config"]["end_factor"],
                        total_iters=self.hparams["train_config"]["warmup_steps"]
                        ),
                    "interval": "step",
                    "name": "learning_rate"
                    }
        elif self.hparams["train_config"]["scheduler"] == "step":    
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [10,20,30], gamma = 0.1, last_epoch=-1
                ),
                "interval": "epoch",
                "frequency":1,
                "name": "learning_rate",
                }
        elif self.hparams["train_config"]["scheduler"] == "linear_step":    
            linear_scheduler = {
                    "scheduler": LinearLR(
                        optimizer,start_factor=self.hparams["train_config"]["start_factor"],
                        end_factor=self.hparams["train_config"]["end_factor"],
                        total_iters=self.hparams["train_config"]["warmup_steps"]
                        ),
                    "interval": "step",
                    "name": "learning_rate"
                    }
            step_scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = [7500,15000,22500], gamma = 0.1, last_epoch=-1
                ),
                "interval": "step",
                "frequency":1,
                "name": "learning_rate",
                }
            scheduler = ChainedScheduler([linear_scheduler,step_scheduler],optimizer=optimizer)
        return scheduler

class SingleSessionModule(Autoencoder_Models):
    """Class for single session autoencoder. 
    """
    def __init__(self,hparams):
        super().__init__(hparams)
        self.model = models[hparams["model"]](hparams["model_config"])
