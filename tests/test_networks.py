"""
Test that networks work as intended. 
"""
import os
import torch
import json
from behavioral_autoencoder.networks import SingleSessionAutoEncoder

here = os.path.abspath(os.path.dirname(__file__))

class Test_SingleSessionAutoEncoder():
    config_path = os.path.join(here,"..","configs","model_configs","alm_default.json")
    def test_init(self):
        with open(self.config_path,"r") as f:
            config = json.load(f)
        ssae = SingleSessionAutoEncoder(config)
    def test_forward(self):
        with open(self.config_path,"r") as f:
            config = json.load(f)
        batch_size=10    
        dummy_input = torch.zeros(10,1,1,config["image_height"],config["image_width"])
        ssae = SingleSessionAutoEncoder(config)
        output = ssae(dummy_input)
        assert output[0].shape == (10,1,1,config["image_height"],config["image_width"])
        assert output[1].shape == (10,1,config["embed_size"])



