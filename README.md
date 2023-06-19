# Audio Augmentation Library for Speech Processing
![Github CI](https://github.com/josebeo2016/audio_augmentor/actions/workflows/python-package.yml/badge.svg)
## [Documentation](https://audio-augmentor.readthedocs.io/en/latest/)
## Install

```
python setup.py install
```

## Quick usage

- Volume up an audio sample to 10dB

```
from audio_augmentor import VolumeAugmentor
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


SAMPLE_WAV = os.path.join(BASE_DIR,"audio.flac")
CONFIG = {
    "aug_type": "volume",
    "output_path": os.path.join(BASE_DIR,"volumeup_audio.flac"),
    "out_format": "flac",
    "min_volume_dBFS": 10,
    "max_volume_dBFS": 10
}
va = VolumeAugmentor(SAMPLE_WAV, CONFIG)
va.run()
assert os.path.exists(os.path.join(CONFIG["output_path"], va.file_name +"."+ CONFIG["out_format"]))

```

* More example could be found in `tests/` or `conditioning.py`