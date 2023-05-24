from audio_augmentor.adversarial import AdversarialNoiseAugmentor
import os
import librosa
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename='test.log')
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_rawnet2_pgd():
    SAMPLE_WAV = os.path.join(BASE_DIR,"data/LA_T_1000137.flac")
    CONFIG = {
        "aug_type": "background_noise",
        "output_path": os.path.join(BASE_DIR,"data/augmented"),
        "out_format": "flac",
        "model_name": "rawnet2",
        "model_pretrained": os.path.join(BASE_DIR,"../pretrained/pre_trained_DF_RawNet2.pth"),
        "config_path": os.path.join(BASE_DIR,"../pretrained/Rawnet2_config.yaml"),
        "device": "cpu",
        "adv_method": "ProjectedGradientDescent",
        "adv_config": {
            "eps": 0.003,
            "eps_step": 0.001,
            "norm": "inf",
        }
    }
    adva = AdversarialNoiseAugmentor(SAMPLE_WAV, CONFIG)
    adva.load()
    adva.transform()
    adva.run()
    # test if the output file exists
    assert os.path.exists(os.path.join(CONFIG["output_path"], adva.file_name +"."+ CONFIG["out_format"]))
    
    # check if the length of file is correct
    ori_y, _ = librosa.load(SAMPLE_WAV, sr=16000)
    aug_y, _ = librosa.load(os.path.join(CONFIG["output_path"], adva.file_name +"."+ CONFIG["out_format"]), sr=16000)
    
    assert len(ori_y) == len(aug_y)
    
    # remove output file
    os.remove(os.path.join(CONFIG["output_path"], adva.file_name +"."+ CONFIG["out_format"]))
    

    
    
