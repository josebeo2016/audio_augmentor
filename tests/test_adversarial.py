from audio_augmentor.adversarial import AdversarialNoiseAugmentor
import os
import librosa
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename='test.log')
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
def down_load_model(model_name):
    if model_name == "rawnet2":
        if os.path.exists(os.path.join(BASE_DIR,"../pretrained/pre_trained_DF_RawNet2.pth")):
            return
        os.system("wget https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip")
        os.system("unzip pre_trained_DF_RawNet2.zip")
        os.system("rm pre_trained_DF_RawNet2.zip")
        os.system("mv pre_trained_DF_RawNet2.pth {}".format(os.path.join(BASE_DIR,"../pretrained/pre_trained_DF_RawNet2.pth")))
    if model_name == "aasistssl":
        if os.path.exists(os.path.join(BASE_DIR,"../pretrained/LA_model.pth")):
            return
        os.system("gdown 11vFBNKzYUtWqdK358_JEygawFzmmaZjO")
        os.system("mv LA_model.pth {}".format(os.path.join(BASE_DIR,"../pretrained/LA_model.pth")))
    
    if model_name == "xlsr2_300m":
        if os.path.exists(os.path.join(BASE_DIR,"../pretrained/xlsr2_300m.pth")):
            return
        os.system("wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt")
        os.system("mv xlsr2_300m.pt {}".format(os.path.join(BASE_DIR,"../pretrained/xlsr2_300m.pth")))
    
        
def test_rawnet2_pgd():
    SAMPLE_WAV = os.path.join(BASE_DIR,"data/LA_T_1000137.flac")
    CONFIG = {
        "aug_type": "adversarial",
        "output_path": os.path.join(BASE_DIR,"data/augmented"),
        "out_format": "flac",
        "model_name": "rawnet2",
        "model_pretrained": os.path.join(BASE_DIR,"../pretrained/pre_trained_DF_RawNet2.pth"),
        "config_path": os.path.join(BASE_DIR,"../pretrained/Rawnet2_config.yaml"),
        "device": "cuda:1",
        "adv_method": "ProjectedGradientDescent",
        "adv_config": {
            "eps": 0.003,
            "eps_step": 0.001,
            "norm": "inf",
        }
    }
    ################################## GITHUB TEST ##################################
    assert True
    ################################## LOCAL TEST ##################################
    # download the pretrained model
    down_load_model(CONFIG["model_name"])
    
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
    
def test_aasistssl_pgd():
    SAMPLE_WAV = os.path.join(BASE_DIR,"data/LA_T_1000137.flac")
    CONFIG = {
        "aug_type": "adversarial",
        "output_path": os.path.join(BASE_DIR,"data/augmented"),
        "out_format": "flac",
        "model_name": "aasistssl",
        "model_pretrained": os.path.join(BASE_DIR,"../pretrained/LA_model.pth"),
        "ssl_model": os.path.join(BASE_DIR,"../pretrained/xlsr2_300m.pth"),
        "device": "cuda:1",
        "adv_method": "ProjectedGradientDescent",
        "adv_config": {
            "eps": 0.003,
            "eps_step": 0.001,
            "norm": "inf",
        }
    }
    ################################## GITHUB TEST ##################################
    assert True
    
    ################################## LOCAL TEST ##################################
    down_load_model(CONFIG["model_name"])
    down_load_model("xlsr2_300m")
    
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
        
        
    
    
