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

    parser.add_argument('--aug_type', type=str, default="background_noise", required=True, help='Augmentation type.\n'
                        +'Suported: background_noise, pitch, speed, volume, reverb, compression, time_stretch, pitch_shift')
    
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
    
def background_noise(args, filename):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "background_noise",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "noise_path": args.noise_path,
        "min_SNR_dB": 5,
        "max_SNR_dB": 15
    }
    bga = BackgroundNoiseAugmentor(in_file, config)
    bga.run()

def pitch(args, filename):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "pitch",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_pitch_shift": -1,
        "max_pitch_shift": 1
    }
    pa = PitchAugmentor(in_file, config)
    pa.run()
    
def reverb(args, filename):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "reverb",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "rir_path": args.rir_path,
    }
    ra = ReverbAugmentor(in_file, config)
    ra.run()

def speed(args, filename):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "speed",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_speed_factor": 0.9,
        "max_speed_factor": 1.1
    }
    sa = SpeedAugmentor(in_file, config)
    sa.run()
    
def volume(args, filename):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "volume",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "min_volume_dBFS": -10,
        "max_volume_dBFS": 10
    }
    va = VolumeAugmentor(in_file, config)
    va.run()

def adversarial(args, filename):
    # load audio:
    in_file = os.path.join(args.input_path, filename)
    config = {
        "aug_type": "adversarial",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "model_name": "rawnet2",
        "model_pretrained": os.path.join(BASE_DIR,"pretrained/pre_trained_DF_RawNet2.pth"),
        "config_path": os.path.join(BASE_DIR,"pretrained/Rawnet2_config.yaml"),
        "device": "cuda:1",
        "adv_method": "ProjectedGradientDescent",
        "adv_config": {
            "eps": 0.003,
            "eps_step": 0.001,
            "norm": "inf",
        }
    }
    ana = AdversarialNoiseAugmentor(in_file, config)
    ana.run()


def main():
    args = parse_argument()
    # prepare config:
    set_start_method("spawn", force=True)
    filenames = os.listdir(args.input_path)
    num_files = len(filenames)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    assert(args.aug_type in SUPPORTED_AUGMENTORS)
    
    func = partial(globals()[args.aug_type], args)
    with Pool(processes=args.thread) as p:
        with tqdm(total=num_files) as pbar:
            for _ in p.imap_unordered(func, filenames):
                pbar.update()

if __name__ == '__main__':
    main()
