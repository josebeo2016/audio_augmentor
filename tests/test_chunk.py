from audio_augmentor.artmodel import lcnn, rawnet2, aasist_ssl
import parse_config
import torch
import os
import librosa
import json
import numpy as np
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# SAMPLE_WAV = os.path.join(BASE_DIR,"data/LA_T_1000137.flac") # short audio
SAMPLE_WAV = os.path.join(BASE_DIR,"data/20230531_161208.wav") # long audio
def test_chunk_lcnn():

    LCNN = lcnn.ArtLCNN(config_path=os.path.join(BASE_DIR,"../pretrained/lcnn_lpsseg_uf_seg600.json"), device=DEVICE)
    
    stft_config = LCNN.config["stft"]
    wav, sr = librosa.load(SAMPLE_WAV, sr=16000)
    print(wav.shape)
    lps = lcnn.extract_LPS(
            x=wav,
            n_fft=stft_config["n_fft"],
            hop_length=stft_config["hop_length"],
            win_length=stft_config["win_length"],
            window=stft_config["window"],
            pre_emphasis=stft_config["pre_emphasis"],
        )
    print(lps.shape)
    lps = np.squeeze(lps)
    lps = lcnn.get_unified_feature(lps, min_n_frame = LCNN.config["arch"]["args"]["min_n_frame"], eval=True)
    lps = np.expand_dims(lps, axis=0)
    print(lps.shape)
    chunk, last_size = LCNN.get_chunk(wav)
    print(chunk[0].size())
    print(last_size)
    
def test_chunk_rawnet2():

    Rawnet2 = rawnet2.ArtRawnet2(config_path=os.path.join(BASE_DIR,"../pretrained/Rawnet2_config"), device=DEVICE)

    wav, sr = librosa.load(SAMPLE_WAV, sr=16000)
    print(wav.shape)

    chunk, last_size = Rawnet2.get_chunk(wav, sr=sr)
    print(chunk[0].size())
    print(Rawnet2.input_shape)
    assert(list(chunk[0].size())==Rawnet2.input_shape)
    print(last_size)
    
def test_chunk_lcnn():

    LCNN = lcnn.ArtLCNN(config_path=os.path.join(BASE_DIR,"../pretrained/lcnn_lpsseg_uf_seg600.json"), device=DEVICE)
    
    stft_config = LCNN.config["stft"]
    wav, sr = librosa.load(SAMPLE_WAV, sr=16000)
    print(wav.shape)
    lps = lcnn.extract_LPS(
            x=wav,
            n_fft=stft_config["n_fft"],
            hop_length=stft_config["hop_length"],
            win_length=stft_config["win_length"],
            window=stft_config["window"],
            pre_emphasis=stft_config["pre_emphasis"],
        )
    print(lps.shape)
    lps = np.squeeze(lps)
    lps = lcnn.get_unified_feature(lps, min_n_frame = LCNN.config["arch"]["args"]["min_n_frame"], eval=True)
    lps = np.expand_dims(lps, axis=0)
    print(lps.shape)
    chunk, last_size = LCNN.get_chunk(wav)
    print(chunk[0].size())
    print(LCNN.input_shape)
    assert(list(chunk[0].size())==LCNN.input_shape)
    print(last_size)



    
    