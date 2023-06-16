from audio_augmentor import TelephoneEncodingAugmentor

import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_run():
    SAMPLE_WAV = os.path.join(BASE_DIR,"data/20230531_161208.wav")
    CONFIG = {
        "aug_type": "telephone",
        "output_path": os.path.join(BASE_DIR,"data/augmented"),
        "out_format": "wav",
        "encoding": "ALAW",
        # bandpass filters can be removed by setting it to None: `"bandpass": None`
        "bandpass": {
            "lowpass": "3400",
            "highpass": "400"
        }
    }
    sa = TelephoneEncodingAugmentor(CONFIG)
    sa.load(SAMPLE_WAV)
    sa.transform()
    sa.save()
    assert os.path.exists(os.path.join(CONFIG["output_path"], sa.file_name +"."+ CONFIG["out_format"]))
    os.remove(os.path.join(CONFIG["output_path"], sa.file_name +"."+ CONFIG["out_format"]))
    