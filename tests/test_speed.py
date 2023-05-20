from audio_augmentor import SpeedAugmentor
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_run():
    SAMPLE_WAV = os.path.join(BASE_DIR,"../test/LA_T_1000137.flac")
    CONFIG = {
        "aug_type": "pitch",
        "output_path": os.path.join(BASE_DIR,"../test/augmented"),
        "out_format": "flac",
        "min_speed_factor": 1.5,
        "max_speed_factor": 1.5
    }
    sa = SpeedAugmentor(SAMPLE_WAV, CONFIG)
    sa.run()
    assert os.path.exists(os.path.join(CONFIG["output_path"], sa.file_name +"."+ CONFIG["out_format"]))
    os.remove(os.path.join(CONFIG["output_path"], sa.file_name +"."+ CONFIG["out_format"]))
    