from .base import BaseAugmentor
from .utils import librosa_to_pydub
import random

import logging
logger = logging.getLogger(__name__)
class VolumeAugmentor(BaseAugmentor):
    def __init__(self, input_path: str, config: dict):
        """
        Volume augmentor class requires these config:
        min_volume_dBFS: float, min volume dBFS
        max_volume_dBFS: float, max volume dBFS
        """
        super().__init__(input_path, config)
        self.volume_dBFS = random.uniform(config["min_volume_dBFS"], config["max_volume_dBFS"])
        
        self.audio_data = None
        
    def load(self):
        # load with librosa
        super().load()
        # transform to pydub audio segment
        self.audio_data = librosa_to_pydub(self.data, sr=self.sr)
        
    def transform(self):
        """
        Volume up or down the audio using pydub `AudioSegment` method
        """
        self.augmented_audio = self.audio_data + self.volume_dBFS