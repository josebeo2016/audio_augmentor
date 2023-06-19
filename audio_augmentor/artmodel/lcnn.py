import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json
from scipy import signal
from torch.nn import Parameter
from torch.autograd import Variable
import librosa
import argparse

############
## ArtModel
## josebeo2016
############

from audio_augmentor.artmodel.artmodel import ArtModelWrapper

class ArtLCNN(ArtModelWrapper):
    def __init__(self, config_path: str, device: str):
        """
        Initialize ArtModelWrapper for LCNN model

        :param config_path: path to config file
        :param device: device to run model on (cpu or cuda)
        """
        super().__init__(device)
        self.model_name = "AasistSSL"
        self.input_shape = [1, 1, 600, 863]
        self.nb_class = 2
        self.device = device
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def load_model(self, model_path: str):
        # load assistssl model

        self.model = LCNN(
            c_s=self.config["arch"]["args"]["c_s"],
            asoftmax=self.config["arch"]["args"]["asoftmax"],
            phiflag=self.config["arch"]["args"]["phiflag"],
        ).to(self.device)
        # load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'], )
        self.model.to(self.device)
        self.model.eval()

    def parse_input(self, input_data: np.ndarray, sr: int = 16000) -> torch.Tensor:
        # get stft config
        stft_conf = self.config["stft"]
        # extract lps
        lps = extract_LPS(
            x=input_data,
            n_fft=stft_conf["n_fft"],
            hop_length=stft_conf["hop_length"],
            win_length=stft_conf["win_length"],
            window=stft_conf["window"],
            pre_emphasis=stft_conf["pre_emphasis"]
        )
        lps = np.squeeze(lps)
        lps = get_unified_feature(lps, min_n_frame = self.config["arch"]["args"]["min_n_frame"],eval=True)
        lps = np.expand_dims(lps, axis=0)
        X = torch.Tensor(lps)
        return X.unsqueeze(0).to(self.device)
    
    def get_chunk(self, input_data: np.ndarray) -> list:
        super().get_chunk(input_data)
        # get stft config
        stft_conf = self.config["stft"]
        min_n_frame = self.config["arch"]["args"]["min_n_frame"]
        # extract lps
        lps = extract_LPS(
            x=input_data,
            n_fft=stft_conf["n_fft"],
            hop_length=stft_conf["hop_length"],
            win_length=stft_conf["win_length"],
            window=stft_conf["window"],
            pre_emphasis=stft_conf["pre_emphasis"],
        )
        chunk_size = len(lps) // min_n_frame
        last_size = len(lps) % min_n_frame
        if chunk_size == 0:
            # need to pad the size to min_n_frame
            temp = get_unified_feature(lps, min_n_frame = min_n_frame,eval=True)
            temp = np.expand_dims(temp, axis=0)
            temp = torch.Tensor(temp).unsqueeze(0).to(self.device)
            
            return [temp], last_size
        else:
            # make a list of chunk which has size of min_n_frame
            chunks = []
            for i in range(chunk_size):
                temp = get_unified_feature(lps[i*min_n_frame:(i+1)*min_n_frame,:], min_n_frame = min_n_frame, eval=True)
                temp = np.expand_dims(temp, axis=0)
                temp = torch.Tensor(temp).unsqueeze(0).to(self.device)
                chunks.append(temp)
            
            if last_size != 0:
                temp = get_unified_feature(lps[-last_size:,:], min_n_frame = min_n_frame, eval=True)
                temp = np.expand_dims(temp, axis=0)
                temp = torch.Tensor(temp).unsqueeze(0).to(self.device)
                chunks.append(temp)
            
            return chunks, last_size
    
    def chunk_to_audio(self, chunks: list, last_size: int) -> np.ndarray:
        # concatenate chunks
        lps = np.concatenate(chunks, axis=1)
        # recover to original size
        if last_size != 0:
            lps = lps[:,:(len(chunks)-1) * self.input_shape[-2] + last_size, :]
        
        lps = np.squeeze(lps)

        # revert back to audio
        stft_conf = self.config["stft"]
        gt_spec = librosa.stft(self.data, n_fft=stft_conf["n_fft"], hop_length=stft_conf["hop_length"], win_length=stft_conf["win_length"], window=stft_conf["window"])
        audio = revert_power_db_to_wav(gt_spec=gt_spec, adv_power_db=lps, n_fft=stft_conf["n_fft"], hop_length=stft_conf["hop_length"], win_length=stft_conf["win_length"], window=stft_conf["window"])
        return audio

    def predict(self, input: np.ndarray):
        """
        return: confidence score of spoof and bonafide class
        """
        self._predict = self.model(self.parse_input(input), eval=True)
        per = nn.Softmax(dim=1)(self._predict)
        _, pred = self._predict.max(dim=1)
        return per[0][0].item() * 100, per[0][1].item() * 100


# Calculate LPS
def preemphasis(wav, k=0.97) -> np.ndarray:
    """
    Pre-emphasis filter on waveform

    :param wav: audio signal
    :param k: pre-emphasis coefficient
    :return: pre-emphasized waveform
    """
    return signal.lfilter([1, -k], [1], wav)

def extract_LPS(
    x: np.ndarray,
    n_fft=512,
    hop_length=160,
    win_length=400,
    window="hamming",
    pre_emphasis=0.97,
    ref=1.0,
    amin=1e-30,
    top_db=None,
) -> np.ndarray:
    """
    Extracts log power spectrogram from audio input

    :param x: audio input
    :param n_fft: FFT window size
    :param hop_length: hop length
    :param win_length: window length
    :param window: window type
    :param pre_emphasis: pre-emphasis coefficient
    :param ref: ref value for log compression
    :param amin: min amplitude for log compression
    :param top_db: max dB for log compression
    :return: log power spectrogram in np.ndarray
    """
    if pre_emphasis is not None:
        x = preemphasis(x, k=pre_emphasis)
    spec = librosa.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
    ).T
    # spec = spec[:-2, :]  # TODO: check why there are two abnormal frames.
    mag_spec = np.abs(spec)
    powspec = np.square(mag_spec)
    logpowspec = librosa.power_to_db(powspec, ref=ref, amin=amin, top_db=top_db)
    return logpowspec


def get_unified_feature(mat: np.ndarray, min_n_frame: int, eval=False):
    """
    If number of frames > min_n_frame, only use the first min_n_frame frames,
    otherwise, pad by repeating and then use the first min_n_frame frames.

    :param mat: has shape [T, D].
    :param min_n_frame: minimum number of frames.
    :param eval: if True, only use the first min_n_frame frames. If False, randomly select a segment of min_n_frame frames.
    :return: unified feature with shape [min_n_frame, D].
    """
    n_frames = mat.shape[0]
    if n_frames == 0:
        return mat

    if n_frames > min_n_frame:
        if not eval:
            ii = np.random.randint(0, n_frames - min_n_frame)
            return mat[ii : (ii + min_n_frame), :]
        else:
            return mat[:min_n_frame, :]

    n_repeat = int(np.ceil(min_n_frame / n_frames))
    mat = np.tile(mat, (n_repeat, 1))
    return mat[:min_n_frame, :]

def power_db_to_mag(power_db: np.ndarray) -> np.ndarray:
    """
    Convert power spectrogram to magnitude spectrogram
    
    :param power_db: power spectrogram in dB
    :return: magnitude spectrogram, in np.ndarray
    """
    power_spec = librosa.core.db_to_power(S_db=power_db, ref=1.0)
    mag_spec = np.sqrt(power_spec) 
    return mag_spec

def revert_power_db_to_wav(gt_spec: np.ndarray, adv_power_db: np.ndarray, n_fft=1724, hop_length=130, win_length=1724, window="blackman") -> np.ndarray:
    """
    Revert audio wavform from power spectrogram (which adversarial attack is applied on) and ground truth spectrogram
    
    :param gt_spec: Tranformed ground truth spectrogram, librosa.stft(gt_wav)
    :param adv_power_db: Power spectrogram of adversarial attack
    :param n_fft: FFT window size
    :param hop_length: hop length
    :param win_length: window length
    :param window: window type
    :return: audio wavform in np.ndarray
    """
    _, phase = librosa.magphase(gt_spec)
    phase = phase[:, :adv_power_db.shape[0]]
    mag = power_db_to_mag(adv_power_db).T
    complex_specgram = mag * phase
    audio = librosa.istft(complex_specgram, hop_length=hop_length, win_length=win_length, window=window)
    return audio

############################
## LCNN model from
## https://github.com/nguyenvulong/AdvAttacksASVspoof/blob/master/model/lcnn.py
############################


### A-softmax
def mypsi(x, m):
    x = x * m
    return (
        1
        - x**2 / math.factorial(2)
        + x**4 / math.factorial(4)
        - x**6 / math.factorial(6)
        + x**8 / math.factorial(8)
        - x**9 / math.factorial(9)
    )


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
        ]

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(
            1e5
        )  # L2-norm along dimension 1, i.e. normalize each column
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())  # size=(B,Classnum)
            k = (
                self.m * theta / 3.14159265
            ).floor()  # theta >= k*pi / m => k <= m*theta/pi
            n_one = k * 0.0 - 1
            psi_theta = (n_one**k) * cos_m_theta - 2 * k  # size=(B,Classnum)
        else:
            theta = cos_theta.acos()
            psi_theta = mypsi(theta, self.m)
            psi_theta = psi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)  # ||x|| * cos_theta
        psi_theta = psi_theta * xlen.view(-1, 1)  # ||x|| * psi_theta
        output = (cos_theta, psi_theta)  # during evaluation, using cos_theta
        # return output  # size=(B,Classnum,2)
        return cos_theta

    def forward_eval(self, input):
        x = input
        w = self.weight

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        # xlen = x.pow(2).sum(1).pow(0.5)
        wlen = ww.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(ww)  # size=(B,Classnum)
        # cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta / wlen.view(1, -1)  # ||x|| * cos_theta
        # cos_theta = cos_theta.clamp(-1,1)
        # cos_theta = cos_theta * xlen.view(-1,1)   # ||x|| * cos_theta

        return cos_theta


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, psi_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()  # to uint8
        index = Variable(index)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += psi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss


###


class mfm(nn.Module):
    """Max-Feature-Map."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        type=1,
        dp_out=0.75,
    ):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(
                in_channels,
                2 * out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            self.filter = nn.Sequential(
                nn.Linear(in_channels, 2 * out_channels), nn.Dropout(p=dp_out)
            )

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class LCNN(nn.Module):
    def __init__(
        self, c_s=[32, 48, 64, 32, 32, 80], asoftmax=True, phiflag=True, num_classes=2
    ):  # inputs shape:[863,600,1]
        super(LCNN, self).__init__()
        # small [8, 12, 12, 4, 4, 32]
        self.c_s = c_s
        self.asoftmax = asoftmax

        self.layer1 = nn.Sequential(
            mfm(1, c_s[0], 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),
        )  # shape[431,300,c_s[0]]

        self.layer2 = nn.Sequential(
            group(c_s[0], c_s[1], 3, 1, 1),
            nn.MaxPool2d(
                kernel_size=2, stride=2, ceil_mode=False
            ),  # shape[215,150,c_s[1]]
            nn.BatchNorm2d(c_s[1]),
        )

        self.layer3 = nn.Sequential(
            group(c_s[1], c_s[2], 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),
        )  # shape[107,75,c_s[2]]

        self.layer4 = nn.Sequential(
            group(c_s[2], c_s[3], 3, 1, 1),
            nn.BatchNorm2d(c_s[3]),
            group(c_s[3], c_s[4], 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),
        )  # shape[53,37,c_s[4]]

        self.fc1 = nn.Sequential(
            mfm(53 * 37 * c_s[4], c_s[5], type=0, dp_out=0.75), nn.BatchNorm1d(c_s[5])
        )

        if self.asoftmax:
            self.fc2 = AngleLinear(c_s[5], num_classes, phiflag=phiflag)
        else:
            self.fc2 = nn.Linear(c_s[5], num_classes)

        self.init_weight()

    def forward(self, x, eval=False):
        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())

        x = x.view(-1, 53 * 37 * self.c_s[4])
        # print(x.size())
        # print('x'*100)
        x = self.fc1(x)

        if eval and self.asoftmax:
            x = self.fc2.forward_eval(x)
        else:
            x = self.fc2(x)

        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(LCNN, self).__str__() + "\nTrainable parameters: {}".format(params)


def lcnn_net(**kwargs):
    model = LCNN(**kwargs)
    return model
