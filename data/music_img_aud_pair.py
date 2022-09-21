
import csv
import glob
import numpy as np
import os
from PIL import Image, ImageEnhance
import random
import scipy.signal
import soundfile as sf

import torch
import torchvision.transforms as transforms


class MusicImgAudPairDataset(object):
    def __init__(self, args, pr, list_sample, split='train'):
        self.pr = pr
        self.split = split
        self.seed = pr.seed
        self.audio_len = args.audio_len
        
        if split == 'train': 
            self.repeat = 100
        else:
            self.repeat = 10
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
        self.video_list = self.list_sample
       
        self.list_sample = self.list_sample * self.repeat

        # shuffle data
        random.seed(self.seed)
        num_sample = len(self.list_sample)
        if self.split == 'train':
            random.shuffle(self.list_sample)

        np.random.seed(1234)
        print('Image-Audio Pair Dataloader: # sample of {}: {}'.format(self.split, num_sample))

    def __getitem__(self, index):
        pass

    def __len__(self): 
        if self.split == 'test': 
            return len([*self.bboxs])
        else: 
            return len(self.list_sample)

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


class MusicImg2AudPairDataset(MusicImgAudPairDataset): 
    def __getitem__(self, index):
        sample = self.list_sample[index]
        
        audio_path = os.path.join(self.pr.data_base_path, sample, 'audio/audio_clips')
        audio_list = glob.glob('%s/*.wav' % audio_path)
        audio_list.sort()

        # load video frames
        frame_folder = os.path.join(self.pr.data_base_path, sample, 'frames')
        frame_list = glob.glob('%s/*.jpg' % frame_folder)
        frame_list.sort()

        total_num_frames = min(len(frame_list), len(audio_list) * self.pr.fps)
        frames_per_sample = int(self.pr.vid_dur * self.pr.fps)

        # randomly select a frame
        if self.split == 'train': 
            rand_frame_index = np.random.choice(np.arange(frames_per_sample, total_num_frames - int(self.audio_len + 2) * frames_per_sample - 1))
        else: 
            rand_frame_index = (total_num_frames - int(self.audio_len) * frames_per_sample - 1) // 2

        frame_info = frame_list[rand_frame_index]
        
        # select corresponding audio
        rand_aud_index = (rand_frame_index + 1) // self.pr.fps
        aud_offset = (rand_frame_index + 1) % self.pr.fps * 1 / self.pr.fps
        
        audio = self.load_aud(audio_list, rand_aud_index, aud_offset)

        # sample another random audio
        if self.split == 'train': 
            rand_frame_index_2 = np.random.choice(np.arange(frames_per_sample, total_num_frames - int(self.audio_len + 2) * frames_per_sample - 1))
        else: 
            rand_frame_index_2 = (total_num_frames -  int(self.audio_len + 2) * frames_per_sample - 1) // 4
        rand_aud_index_2 = (rand_frame_index_2 + 1) // self.pr.fps
        aud_offset_2 = (rand_aud_index_2 + 1) % self.pr.fps * 1 / self.pr.fps
        audio_2 = self.load_aud(audio_list, rand_aud_index_2, aud_offset_2)

        vision_dict = self.process_video([frame_info])

        label = 0

        batch = {
            'audio_info': audio_path,
            'frame_info': frame_info, 
            'frames': vision_dict['frameset'],
            'audio': audio,
            'audio_2': audio_2, 
            'label': label
        }
        return batch
    
    def load_aud(self, audio_list, rand_aud_index, aud_offset): 
        if aud_offset == 0: 
            if self.audio_len == 1: 
                audio_path = audio_list[rand_aud_index]
                audio, audio_rate = sf.read(audio_path, dtype='float64')
            elif self.audio_len == 3: 
                audio_path_0 = audio_list[rand_aud_index - 1]
                audio_path_1 = audio_list[rand_aud_index]
                audio_path_2 = audio_list[rand_aud_index + 1]
                audio_0, audio_rate = sf.read(audio_path_0, dtype='float64')
                audio_1, audio_rate = sf.read(audio_path_1, dtype='float64')
                audio_2, audio_rate = sf.read(audio_path_2, dtype='float64')

                audio = np.concatenate((audio_0, audio_1, audio_2), axis=0)
        
        else:
            if self.audio_len == 1:  
                audio_path_0 = audio_list[rand_aud_index - 1]
                audio_path_1 = audio_list[rand_aud_index]
                audio_0, audio_rate = sf.read(audio_path_0, dtype='float64')
                audio_1, audio_rate = sf.read(audio_path_1, dtype='float64')

                audio_0_clip = audio_0[- int(audio_rate * (1 - aud_offset)):]
                audio_1_clip = audio_1[: int(audio_rate * aud_offset)]
                audio = np.concatenate((audio_0_clip, audio_1_clip), axis=0)
            elif self.audio_len == 3: 
                audio_path_0 = audio_list[rand_aud_index - 1]
                audio_path_1 = audio_list[rand_aud_index]
                audio_path_2 = audio_list[rand_aud_index + 1]
                audio_path_3 = audio_list[rand_aud_index + 2]
                audio_0, audio_rate = sf.read(audio_path_0, dtype='float64')
                audio_1, audio_rate = sf.read(audio_path_1, dtype='float64')
                audio_2, audio_rate = sf.read(audio_path_2, dtype='float64')
                audio_3, audio_rate = sf.read(audio_path_3, dtype='float64')

                audio_0_clip = audio_0[- int(audio_rate * (1 - aud_offset)):]
                audio_3_clip = audio_3[: int(audio_rate * aud_offset)]
                audio = np.concatenate((audio_0_clip, audio_1, audio_2, audio_3_clip), axis=0)
        
        audio = np.mean(audio, axis=-1)
        sample_len = int(self.pr.vid_dur * self.audio_len * audio_rate)
        audio = audio[:sample_len]

        audio = scipy.signal.resample(audio, num=int(len(audio) / audio_rate * self.pr.samp_sr), axis=-1)
        audio = self.normalize_audio(audio)
        audio = torch.from_numpy(audio.copy()).float()
        return audio