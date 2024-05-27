from audio_augmentor.background_noise import BackgroundNoiseAugmentor
from audio_augmentor.pitch import PitchAugmentor
from audio_augmentor.reverb import ReverbAugmentor
from audio_augmentor.speed import SpeedAugmentor
from audio_augmentor.volume import VolumeAugmentor
from audio_augmentor.telephone import TelephoneEncodingAugmentor

# from audio_augmentor.adversarial import AdversarialNoiseAugmentor

# from audio_augmentor import artmodel

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

import logging.config
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "std": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
        }
    },
    "handlers": {
        "default": {
            "class": "logging.NullHandler",
        },
        "test": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.INFO,
        },
    },
    "loggers": {
        "art": {"handlers": ["default"]},
        "tests": {"handlers": ["test"], "level": "INFO", "propagate": True},
    },
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)