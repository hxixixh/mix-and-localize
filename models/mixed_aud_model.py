import torch
import torch.nn as nn
import torchaudio

from . import sourcesep
from .base_model import resnet18


class MixAudModelFeat(nn.Module): 
    def __init__(self, pr, num_node=2): 
        super(MixAudModelFeat, self).__init__()
        self.aud_model = resnet18(modal='audio')
        self.vision_model = resnet18(modal='vision', pretrained=True)
        self.num_node = num_node
        self.pr = pr

        self.stft = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, 
                n_fft=160, 
                win_length=160,
                hop_length=80, 
                n_mels=64)
        )
        # visual ops
        self.conv_v_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_v_2 = nn.Conv2d(128, 128, kernel_size=1)

        # audio ops
        self.pooling_a = nn.AdaptiveMaxPool2d((1, 1))
        for i in range(num_node): 
            fc_a = nn.Sequential(
                nn.Linear(512, 128), 
                nn.ReLU(), 
                nn.Linear(128, 128)
            )
            setattr(self, "fc_a_%d" % (i+1), fc_a)
        self.conv_av = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        
        # fusion ops
        self.max_pooling_av = nn.AdaptiveMaxPool2d((1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, img, aud): 
        pass

    def audio_stft(self, audio):
        if len(audio.shape) == 2: 
            audio = audio.unsqueeze(1) 
        N, C, A = audio.size()
        
        audio = audio.view(N * C, A)
        audio = self.stft(audio)
        audio = audio.transpose(-1, -2)
        audio = sourcesep.db_from_amp(audio, cuda=True)
        audio = sourcesep.normalize_spec(audio, self.pr)
        _, T, F = audio.size()
        audio = audio.view(N, C, T, F)
        return audio

class MixAudModelFeatMultiAud(MixAudModelFeat): 
    def forward(self, img, aud):
        B = img.shape[0]
        if len(aud.shape) != 4: 
            specg = self.audio_stft(aud)
        else: 
            specg = aud
        
        # image features
        feat_img = self.vision_model(img)
        v = self.conv_v_1(feat_img)
        v = self.relu(v)
        v = self.conv_v_2(v)
        v = torch.nn.functional.normalize(v, dim=1)

        # audio features 
        feat_aud = self.aud_model(specg)
        feat_aud = self.pooling_a(feat_aud)
        feat_aud = torch.flatten(feat_aud, 1)
        av_scores = []
        av_maps = []
        a_s = []
        for i in range(self.num_node): 
            a = getattr(self, "fc_a_%d" % (i+1))(feat_aud)
            a = torch.unsqueeze(torch.unsqueeze(a, -1), -1)
            a = torch.nn.functional.normalize(a, dim=1)
            a_s.append(a)
        
        # reshape v features to [B, 128, 14, 14] to [B * 14 * 14, 128]
        v_ = v.permute(0, 2, 3, 1).reshape(-1, 128) 
        a_ = torch.stack(a_s, dim=1).squeeze().reshape(-1, 128, 1)

        av_sim = (v_ @ a_.T).reshape(v.shape[0], 14, 14, a_.shape[0])  # [256v, 14, 14, 256a]
        av_sim = av_sim.permute(0, 3, 1, 2).reshape(-1, 1, 14, 14)     # [256 * 256 v*a, 1, 14, 14]
        av_sim_biased = self.conv_av(av_sim)

        av_score = av_sim_biased.reshape(v.shape[0], a_.shape[0], 14, 14)  # [v, a, 14, 14]
        if v.shape[0] != a_.shape[0]:
            av_map_1 = av_score[range(v.shape[0]), range(v.shape[0])]
            av_map_2 = av_score[:, v.shape[0]:][range(v.shape[0]), range(v.shape[0])]
            av_map = torch.cat([av_map_1, av_map_2], dim=0)
        
        else: 
            av_map = av_score[range(v.shape[0]), range(v.shape[0])]
        
        return av_score, av_map, v, a_s