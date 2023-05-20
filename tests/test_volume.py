from audio_augmentor import VolumeAugmentor
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_run():
    SAMPLE_WAV = os.path.join(BASE_DIR,"../test/LA_T_1000137.flac")
    CONFIG = {
        "aug_type": "volume",
        "output_path": os.path.join(BASE_DIR,"../test/augmented"),
        "out_format": "flac",
        "min_volume_dBFS": -10,
        "max_volume_dBFS": 10
    }
    va = VolumeAugmentor(SAMPLE_WAV, CONFIG)
    va.run()
    assert os.path.exists(os.path.join(CONFIG["output_path"], va.file_name +"."+ CONFIG["out_format"]))
    os.remove(os.path.join(CONFIG["output_path"], va.file_name +"."+ CONFIG["out_format"]))