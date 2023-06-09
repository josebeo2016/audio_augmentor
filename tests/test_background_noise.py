from audio_augmentor import BackgroundNoiseAugmentor
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
def test_run():
    SAMPLE_WAV = os.path.join(BASE_DIR,"data/LA_T_1000137.flac")
    CONFIG = {
        "aug_type": "background_noise",
        "output_path": os.path.join(BASE_DIR,"data/augmented"),
        "out_format": "flac",
        "noise_path": os.path.join(BASE_DIR,"data/musan_sample"),
        "min_SNR_dB": 0,
        "max_SNR_dB": 20
    }
    bga = BackgroundNoiseAugmentor(CONFIG)
    bga.load(SAMPLE_WAV)
    bga.transform()
    bga.save()
    assert os.path.exists(os.path.join(CONFIG["output_path"], bga.file_name +"."+ CONFIG["out_format"]))
    os.remove(os.path.join(CONFIG["output_path"], bga.file_name +"."+ CONFIG["out_format"]))
    