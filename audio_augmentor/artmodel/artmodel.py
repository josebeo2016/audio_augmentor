from art.estimators.classification import PyTorchClassifier
import torch
import numpy as np

class ArtModelWrapper():
    def __init__(self, device):
        self.model = None
        self.nb_class = None
        self.model_name = None
        self.input_shape = None
        self.device = device
        
    def load_model(self, model_path: str):
        raise NotImplementedError
        
    def get_art(self):
        classifier_art = PyTorchClassifier(
            model=self.model,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=None,
            input_shape=self.input_shape,
            nb_classes=self.nb_class,
            clip_values=(-2**15, 2**15 - 1),
            device = self.device
        )
        return classifier_art
    
    def get_chunk(self, input_data: np.ndarray) -> list:
        # save original audio data
        self.data = input_data
    
    def chunk_to_audio(self, chunks: list, last_size: int) -> np.ndarray:
        raise NotImplementedError
    
    def parse_input(self, input_data, sr: int = 16000):
        raise NotImplementedError
    
    def predict(self, input: np.ndarray):
        # make sure model eval mode
        self.model.eval()
        self._predict = self.model(self.parse_input(input))
        
    def batch_load(self, input_dir: str, sr: int = 16000):
        raise NotImplementedError
        