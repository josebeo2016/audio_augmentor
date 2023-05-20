from audio_augmentor import utils
import os
import librosa
import numpy as np
from pydub import AudioSegment
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_pydub_to_librosa():
    SAMPLE_WAV = os.path.join(BASE_DIR,"../test/LA_T_1000137.flac")
    a = AudioSegment.from_file(SAMPLE_WAV)
    b, _ = librosa.load(SAMPLE_WAV, sr=16000)
    
    b_p = np.array(b* (1<<15), dtype=np.int16)
    
    c_a = utils.pydub_to_librosa(a)
    # c_b = utils.librosa_to_pydub(b)
    assert(c_a==b_p).all()
    
def test_librosa_to_pydub():
    SAMPLE_WAV = os.path.join(BASE_DIR,"../test/LA_T_1000137.flac")
    a = AudioSegment.from_file(SAMPLE_WAV)
    b, _ = librosa.load(SAMPLE_WAV, sr=16000)
    
    c_b = utils.librosa_to_pydub(b)
    
    assert(c_b==a)