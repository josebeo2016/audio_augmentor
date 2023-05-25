from audio_augmentor.background_noise import BackgroundNoiseAugmentor
from audio_augmentor.pitch import PitchAugmentor
from audio_augmentor.reverb import ReverbAugmentor
from audio_augmentor.speed import SpeedAugmentor
from audio_augmentor.volume import VolumeAugmentor

from audio_augmentor.adversarial import AdversarialNoiseAugmentor

from audio_augmentor import artmodel

# from . import utils

from .__version__ import (
    
    __author__,
    __author_email__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
)

SUPPORTED_AUGMENTORS = ['background_noise', 'pitch', 'speed', 'volume', 'reverb', 'adversarial']