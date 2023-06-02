import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter

from audio_augmentor.artmodel.artmodel import ArtModelWrapper
from audio_augmentor.artmodel.btse_model.model_one import RawNet
import yaml
import librosa

############
## ArtModel
## josebeo2016
## Based on: https://github.com/josebeo2016/BTS-Encoder-ASVspoof/tree/main/asvspoof2021/LA/Baseline-RawNet2-bio
############

class ArtBTSE(ArtModelWrapper):
    def __init__(self, config_path: str, device: str):
        super().__init__(device)
        self.model_name = "rawnet2"
        self.input_shape = [1, 64600]
        self.nb_class = 2
        self.device = device
        self.config_path = config_path
    
    def load_model(self, model_path: str):
        # load rawnet2 config
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        # load rawnet2 model
        self.model = RawNet(config['model'],device=self.device).to(self.device)
        # load rawnet2 weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def parse_input(self, input_data: np.ndarray, sr: int = 16000) -> torch.Tensor:
        X_pad= pad(input_data, 64600)
        X_pad = Tensor(X_pad)
        return X_pad.unsqueeze(0).to(self.device)
    
    def get_chunk(self, input_data: np.ndarray, sr: int=16000):
        chunk_size = len(input_data) // self.input_shape[1]
        last_size = len(input_data) % self.input_shape[1]
        chunks = []
        if chunk_size == 0:
            # return the parsed input of the redundant
            return [self.parse_input(input_data)], last_size
        for i in range(chunk_size):
            temp = input_data[i* self.input_shape[1] : (i + 1) * self.input_shape[1]]
            temp = self.parse_input(temp)
            chunks.append(temp)
        if last_size != 0:
            chunks.append(self.parse_input(input_data[-last_size:]))
        return chunks, last_size
    
    def chunk_to_audio(self, chunks: list, last_size: int) -> np.ndarray:
        # concatenate chunks
        res = np.concatenate(chunks, axis=0)
        if last_size == 0:
            return res
        else:
            return res[:(len(chunks)-1) * self.input_shape[1] + last_size]
    
    def predict(self, input: np.ndarray):
        """
        return: confidence score of spoof and bonafide class
        """
        super().predict(input)
        per = nn.Softmax(dim=1)(self._predict)
        _, pred = self._predict.max(dim=1)
        return per[0][0].item()*100, per[0][1].item()*100
    


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x
