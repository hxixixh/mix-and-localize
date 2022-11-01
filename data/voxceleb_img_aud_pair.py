import csv
import glob
import numpy as np
import os
from PIL import Image
import random
import scipy.signal
import soundfile as sf

import torch
import torchvision.transforms as transforms

class VoxCelebImg2AudPairDataset(object):
    def __init__(self, args, pr, list_sample, split='train'): 
        self.pr = pr
        self.split = split
        self.seed = pr.seed

        self.repeat = args.repeat
        self.max_sample = args.max_sample

        if split == 'train': 
            vision_transform_list = [
                transforms.Resize((self.pr.img_size, self.pr.img_size)),
                transforms.ToTensor(), 
                transforms.RandomHorizontalFlip(), 
                transforms.RandomResizedCrop((self.pr.img_size, self.pr.img_size), scale=(0.8, 1.0),  ratio=(0.8, 1.2)), 
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
        else: 
            vision_transform_list = [
                transforms.Resize((self.pr.img_size, self.pr.img_size)),
                transforms.ToTensor(), 
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
        self.vision_transform = transforms.Compose(vision_transform_list)

        # load sample list
        if isinstance(list_sample, str):
            self.list_sample = []
            csv_file = csv.reader(open(list_sample, 'r'), delimiter=' ')
            for row in csv_file:
                self.list_sample.append(row[0])

        if self.max_sample > 0: 
            self.list_sample = self.list_sample[0:self.max_sample]   
        self.list_sample = self.list_sample * self.repeat
        
        self.speaker_ids = {}
        for i in range(len(self.list_sample)): 
            p_id = self.list_sample[i].split('/')[-2]
            if p_id not in self.speaker_ids.keys(): 
                self.speaker_ids[p_id] = []
            self.speaker_ids[p_id].append(self.list_sample[i])
        self.list_sample_speaker = list(self.speaker_ids.keys())

        self.list_sample_speaker = self.list_sample_speaker * (len(self.list_sample) // len(self.list_sample_speaker))
        random.seed(self.seed)
        num_sample = len(self.list_sample)
        if self.split == 'train':
            random.shuffle(self.list_sample)

        np.random.seed(1234)
        frames_per_sample = int(self.pr.vid_dur * self.pr.fps)
        if self.split in ['val', 'test']: 
            start_point = 0
            self.val_rand = (start_point * np.ones(num_sample)).astype(int)
            self.val_rand_2 = (frames_per_sample // 2 * np.ones(num_sample)).astype(int)
        print('Image-Audio Pair Dataloader: # sample of {}: {}'.format(self.split, num_sample))

    def __getitem__(self, index):
        sample_id = self.list_sample_speaker[index]
        sample_clips = self.speaker_ids[sample_id]
        sample = random.choice(sample_clips)

        audio_path = os.path.join(sample, 'audio', 'audio.wav')
        audio, audio_rate = sf.read(audio_path, dtype='float64')
        if self.pr.mono and len(audio.shape) == 2: 
            audio = np.mean(audio, axis=-1)

        # load video frames
        frame_folder = os.path.join(sample, 'frames')
        frame_list = glob.glob('%s/*.jpg' % frame_folder)
        frame_list.sort()
        frames_per_sample = int(self.pr.vid_dur * self.pr.fps)

        # randomly select a frame
        if self.split == 'train': 
            total_length = min(len(frame_list), int(audio.shape[0] / audio_rate * self.pr.fps))
            assert (total_length - frames_per_sample - 1) > 0, "video is {}, shape is {}, total:{}, frame per sample:{}".format(audio_path, audio.shape, total_length, frames_per_sample)
            rand_start = np.random.choice(total_length - 3*frames_per_sample - 1)
        else: 
            rand_start = self.val_rand[index]

        frame_info = frame_list[rand_start]

        # select corresponding audio
        sample_len = int(self.pr.vid_dur * audio_rate)
        audio_start = int(rand_start / self.pr.fps * audio_rate)
        audio_0 = audio[audio_start: audio_start + sample_len]
        assert audio_0.shape[0] == sample_len, "{} is broken".format(sample)
        audio_0 = scipy.signal.resample(audio_0, num=int(len(audio_0) / audio_rate * self.pr.samp_sr), axis=-1)
        audio_0 = self.normalize_audio(audio_0)
        audio_0 = torch.from_numpy(audio_0.copy()).float()

        # sample anotehr random audio
        sample_2 = random.choice(self.speaker_ids[sample_id])
        audio_path_2 = os.path.join(sample_2, 'audio', 'audio.wav')
        audio_2, audio_rate = sf.read(audio_path_2, dtype='float64')
        if self.pr.mono and len(audio_2.shape) == 2: 
            audio_2 = np.mean(audio_2, axis=-1)
        sample_len = int(self.pr.vid_dur * audio_rate)
        audio_start_2 = random.choice(range(sample_len - frames_per_sample - 1))
        audio_1 = audio_2[audio_start_2: audio_start_2 + sample_len]
        assert audio_1.shape[0] == sample_len, "{} is broken".format(sample)
        audio_1 = scipy.signal.resample(audio_1, num=int(len(audio_1) / audio_rate * self.pr.samp_sr), axis=-1)
        audio_1 = self.normalize_audio(audio_1)
        audio_1 = torch.from_numpy(audio_1.copy()).float()

        # Get tensors
        vision_dict = self.process_video([frame_info])
        
        batch = {
            'audio_info': audio_path,
            'frame_info': frame_info, 
            'frames': vision_dict['frameset'],
            'audio': audio_0, 
            'audio_2': audio_1, 
        }
        return batch

    def __len__(self): 
        return len(self.list_sample_speaker)

    def process_video(self, image_list):
        frame_list = []
        for image in image_list:
            image = Image.open(image).convert('RGB')
            image = self.vision_transform(image)
            frame_list.append(image.unsqueeze(1))
        frame_list = torch.cat(frame_list, dim=1).squeeze()
        return {
            'frameset': frame_list
        }

    def normalize_audio(self, samples, desired_rms=0.1, eps=1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
        samples = samples * (desired_rms / rms)
        return samples