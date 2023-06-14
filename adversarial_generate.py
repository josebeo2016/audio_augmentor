# import required module
import os
from multiprocessing import Pool, set_start_method
import argparse
from tqdm import *
from functools import partial
import logging
import librosa
import soundfile as sf
from random import randrange
import torch
from pydub import AudioSegment
import random

from audio_augmentor import BackgroundNoiseAugmentor, PitchAugmentor, ReverbAugmentor, SpeedAugmentor, VolumeAugmentor, AdversarialNoiseAugmentor
from audio_augmentor import SUPPORTED_AUGMENTORS
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Set up logging
# logging.basicConfig(filename='running.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')


def parse_argument():
    parser = argparse.ArgumentParser(
        epilog=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
        

    parser.add_argument('--thread', type=int, default=16, help='Number of threads')

    parser.add_argument('--input_path', type=str, default="",required=True, help='Audio file path')
    
    parser.add_argument('--output_path', type=str, default="",required=True, help='Feature output path')
    
    parser.add_argument('--out_format', type=str, default="flac", required=False, help='Output format. \n'
                        +'Suported: flac, ogg, mp3, wav. Default: flac. \n'
                        +'Encode by pydub + ffmpeg. Please install ffmpeg first. \n')

    # env_noise
    parser.add_argument('--noise_path', type=str, default="./musan/",required=False, help='Noise file path')
    
    # reverb
    parser.add_argument('--rir_path', type=str, default="./RIRS_NOISES/simulated_rirs",required=False, help='Reverb IR file path')
    
    # load argument
    args = parser.parse_args()
        
    return args
   

def main():
    args = parse_argument()
    # prepare config:
    filenames = os.listdir(args.input_path)
    num_files = len(filenames)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    config = {
            "aug_type": "adversarial",
            "output_path": args.output_path,
            "out_format": args.out_format,
            "model_name": "btse",
            "model_pretrained": "pretrained/tts_vc_trans_64_concat.pth",
            "config_path": "pretrained/model_config_RawNet_Trans_64concat.yaml",
            "device": "cuda",
            "adv_method": "ProjectedGradientDescent",
            "adv_config": {
                "eps": 0.003,
                "eps_step": 0.001,
                "norm": "inf",
            }
        }
    ana = AdversarialNoiseAugmentor(config)
    for filename in tqdm(filenames):
        print(filename)
        in_file = os.path.join(args.input_path, filename)
        # More model could be found at AISRC1 ls /data/longnv/_saved/models/LA_lcnn_LPSseg_uf_seg600/
        # or /datab/PretrainedModel/adversarial_model_longnv 
        # Note that, the stft config should be add into the original config file before using this code. This make the code solid.
        ana.load(in_file)
        ana.transform()
        ana.save()

if __name__ == '__main__':
    main()
