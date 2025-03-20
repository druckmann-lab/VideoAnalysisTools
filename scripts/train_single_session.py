"""Train a single session autoencoder on provided data. 

Saves checkpoints for the corresponding model, outputs to tensorboard, and finally dumps all predictions and latents into a save directory. 

"""
import os
import fire
import json
import joblib
import datetime
import pytorch_lightning as pl
from behavioral_autoencoder.module import SingleSessionModule
from behavioral_autoencoder.dataset import CropResizeProportion
from behavioral_autoencoder.dataloading import SessionFramesDataModule
from behavioral_autoencoder.eval import get_all_predicts_latents,get_dl_predicts_latents
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

here = os.path.join(os.path.abspath(os.path.dirname(__file__)))

def main(model_config_path, train_config_path, data_path, data_config_path):
    """This main function takes as input four paths. These paths indicate the model configuration parameters, training configuration parameters, path to the data directory, and cropping configuration, respectively. By default we assume that we are training a single session autoencoder.  
    """
    print("\n=== Starting Single Session Autoencoder Training ===")

    ## Model setup 
    print("\nLoading configurations...")
    with open(model_config_path,"r") as f:
        model_config = json.load(f)
    with open(train_config_path,"r") as f:
        train_config = json.load(f)
    model_name = "single_session_autoencoder"

    print(f"Model config: {model_config}")
    print(f"Training config: {train_config}")

    hparams = {
            "model":model_name,
            "model_config":model_config,
            "train_config":train_config
            }

    print("\nInitializing model...")
    ssm = SingleSessionModule(hparams)

    ## Data setup 
    with open(data_config_path,"r") as f:
        data_process_config = json.load(f)
    print("\nSetting up data...")
    alm_cropping = CropResizeProportion(data_config_path)
    data_config = {
            "data_path":data_path,
            "transform":alm_cropping,
            "data_config_path":data_config_path,
            "extension":data_process_config["extension"],
            "trial_pattern":data_process_config["trial_pattern"]
            }
    print(f"Data config: {data_config}")

    print("Initializing data module...")
    sfdm = SessionFramesDataModule(
            data_config,
            train_config["batch_size"],
            train_config["num_workers"],
            train_config["subsample_rate"],
            train_config["subsample_offset"],
            train_config["val_subsample_rate"],
            train_config["val_subsample_offset"]
            )

    ## Set up logging and trainer
    print("\nSetting up logging and checkpoints...")
    date=datetime.datetime.now().strftime("%m-%d-%y")
    time=datetime.datetime.now().strftime("%H_%M_%S")
    timestamp_model = os.path.join(here,"models",model_name,date,time)
    timestamp_pred = os.path.join(here,"preds",model_name,date,time)
    print(f"Model will be saved to: {timestamp_model}")
    print(f"Predictions will be saved to: {timestamp_pred}")

    logger = TensorBoardLogger("tb_logs",name="test_single_session_auto",log_graph=True)
    checkpoint = ModelCheckpoint(monitor="mse/val", mode="min", save_last=True, dirpath=timestamp_model)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    print("\nInitializing trainer...")
    trainer = pl.Trainer(
        fast_dev_run=train_config["fast_dev_run"],
        max_epochs=train_config["max_epochs"],
        accelerator=train_config["accelerator"],
        enable_checkpointing=True,
        callbacks=[checkpoint,lr_monitor],
        log_every_n_steps=1,
        logger=logger,
        enable_progress_bar=True,
    )

    ## Fit the model
    print(f"\nStarting training for {train_config['max_epochs']} epochs...")
    trainer.fit(ssm,sfdm)
    print("Training completed!")

    ## Get out predictions 
    print("\nGenerating predictions and latents...")
    preds,latents = get_dl_predicts_latents(ssm,sfdm.val_dataloader(),sfdm.mean_image,train_config["batch_size"],train_config["num_workers"])

    ## Save out all relevant metadata
    print("\nSaving results...")
    os.makedirs(timestamp_pred, exist_ok=True)
    joblib.dump(preds,os.path.join(timestamp_pred,"preds"))
    joblib.dump(latents,os.path.join(timestamp_pred,"latents"))
    with open(os.path.join(timestamp_pred,"model_config"),"w") as f:
        json.dump(hparams, f)
    with open(os.path.join(timestamp_pred,"data_config"),"w") as f:
        json.dump(data_config, f)

    print("\n=== Training Complete ===")
    print(f"Results saved to: {timestamp_pred}")

if __name__ == "__main__":
    fire.Fire(main)
