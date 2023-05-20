import os
import numpy as np
import librosa  
from pydub import AudioSegment

def recursive_list_files(path, file_type=["wav", "mp3", "flac"]):
    """Recursively lists all files in a directory and its subdirectories"""
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            real_file_type = filename.split(".")[-1]
            if (real_file_type in file_type):
                files.append(os.path.join(dirpath, filename))
    return files

def pydub_to_librosa(audio_segment):
    """Convert pydub audio segment to librosa audio data"""
    return np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

def librosa_to_pydub(audio_data, sr=16000):
    """Convert librosa audio data to pydub audio segment"""
    audio_data = np.array(audio_data * (1<<15), dtype=np.int16)
    return AudioSegment(audio_data.tobytes(), 
                    frame_rate=sr,
                    sample_width=audio_data.dtype.itemsize, 
                    channels=1)
    