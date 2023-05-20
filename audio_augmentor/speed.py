from .base import BaseAugmentor
from .utils import librosa_to_pydub
from pydub import AudioSegment


class SpeedAugmentor(BaseAugmentor):
    def __init__(self, input_path, config):
        """
        Speed augmentor class requires these config:
        speed_factor: float, speed factor
        """
        super().__init__(input_path, config)
        self.speed_factor = config["speed_factor"]
        self.audio_data = None
        
    def load(self):
        # load with librosa
        super().load()
        # transform to pydub audio segment
        self.audio_data = librosa_to_pydub(self.data, sr=self.sr)
        
    def transform(self):
        self.augmented_audio = self.audio_data.speedup(self.speed_factor)