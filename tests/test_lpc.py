from audio_augmentor.artmodel import lcnn
import parse_config

import os
import librosa
import json
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_extract_lpc():
    SAMPLE_WAV = os.path.join(BASE_DIR,"data/LA_T_1000137.flac")

    LCNN = lcnn.ArtLCNN(config_path=os.path.join(BASE_DIR,"../pretrained/lcnn_lpsseg_uf_seg600.json"), device="cpu")
    # load audio with librosa
    wav, sr = librosa.load(SAMPLE_WAV, sr=16000)
    lpc = LCNN.parse_input(wav, sr=sr)

    assert(list(lpc.size())==LCNN.input_shape)
    # Load model
    # This model used https://github.com/victoresque/pytorch-template for developing. Therefore, the model saved its config, including parse_config module.
    # To load the model, we need to import the parse_config.py file as the example in this test file.
    # The parse_config.py file is copied from the original repo and modified to fit this project.
    LCNN.load_model(os.path.join(BASE_DIR,"../pretrained/lcnn_full_230209.pth"))
    print(LCNN.predict(wav))

