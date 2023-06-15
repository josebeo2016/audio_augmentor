from .base import BaseAugmentor
from .utils import recursive_list_files, librosa_to_pydub
import scipy.signal as ss
import librosa
import random
import numpy as np
import torchaudio.functional as F
import logging
import torch

logger = logging.getLogger(__name__)
class TelephoneEncodingAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        super().__init__(config)
        self.encoding = config.get("encoding", "ULAW")
        
    def transform(self):
        """
        """
        self.augmented_audio = librosa_to_pydub(F.apply_codec(torch.tensor(self.data).reshape(1, -1), self.sr, "wav", encoding=self.encoding))