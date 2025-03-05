"""Train a single session autoencoder on provided data. 

Saves checkpoints for the corresponding model, outputs to tensorboard, and finally dumps all predictions and latents into a save directory. 

"""
import os
import fire
import joblib
import datetime
from behavioral_autoencoder.module import SingleSessionModule
from behavioral_autoencoder.dataset import CropResizeProportion
from behavioral_autoencoder.dataloading import SessionFramesDataModule
from behavioral_autoencoder.eval import get_all_predicts_latents
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

here = os.path.join(os.path.abspath(os.path.dirname(__file__)))

def main(model_config_path,train_config_path,data_path,crop_config_path):
    """This main function takes as input four paths. These paths indicate the model configuration parameters, training configuration parameters, path to the data directory, and cropping configuration, respectively. By default we assume that we are training a single session autoencoder.  
    """
    ## Model setup 
    with open(model_config_path,"r") as f:
        model_config = json.load(f)
    with open(train_config_path,"r") as f:
        train_config = json.load(f)
    model_name = "single_session_autoencoder"    

    hparams = {
            "model":model_name,
            "model_config":model_config,
            "train_config":train_config
            }

    ssm = SingleSessionModule(hparams)

    ## Data setup 
    alm_cropping = CropResizeProportion(crop_config_path)
    data_config = {
            "data_path":session_dir,
            "transform":alm_cropping,
            "extension":".png",
            "trial_pattern":None
            }
    sfdm = SessionFramesDataModule(
            data_config,
            train_config["batch_size"],
            train_config["num_workers"],
            train_config["subsample_rate"],
            train_config["subsample_offset"])

    ## Set up logging and trainer
    date=datetime.datetime.now().strftime("%m-%d-%y")
    time=datetime.datetime.now().strftime("%H_%M_%S")
    timestamp_model = os.path.join(here,"models",model_name,date,time)
    timestamp_pred = os.path.join(here,"preds",pred_name,date,time)
    logger = TensorBoardLogger("tb_logs",name="test_single_session_auto",log_graph=True)

    checkpoint = ModelCheckpoint(monitor="acc/mse", mode="max", save_last=False, dirpath = outdir)

    trainer = pl.Trainer(
        fast_dev_run=train_config["fast_dev_run"],
        max_epochs=train_config["max_epochs"],
        accelerator=train_config["accelerator"],  # Explicitly use CPU for testing
        enable_checkpointing=True,  # Disable for testing
        checkpoint_callback = checkpoint,
        log_every_n_steps = 1,
        logger=logger,  # Disable logging for testing
        enable_progress_bar=True,  # See training progress
    )

    ## Fit the model
    trainer.fit(ssm,sfdm)

    ## Get out predictions 
    preds,latents = get_all_predicts_latents(ssm,sfdm,train_config["batch_size"],train_config["num_workers"])

    ## Save out all relevant metadata. 
    os.mkdir(timestamp_pred)
    joblib.dump(preds,os.path.join(timestamp_pred,"preds"))
    joblib.dump(latents,os.path.join(timestamp_pred,"latents"))
    with open(os.path.join(timestamp_pred,"model_config"),"w") as f:
        json.dump(hparams)

    with open(os.path.join(timestamp_pred,"data_config"),"w") as f:
        json.dump(data_config)


if __name__ == "__main__":
    fire.Fire(main)
