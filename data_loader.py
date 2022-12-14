import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa    
import random

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, max_seq_len, subjects_dict, args, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.max_seq_len = max_seq_len
        if(args.facebook):
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        else:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained('TencentGameMate/chinese-wav2vec2-base')

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]

        if(random.uniform(0, 1)>0.7):
            audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=random.uniform(-3, 3))
        if(random.uniform(0, 1)>0.7):
            random_values = np.random.rand(len(audio))
            audio += 0.01*random_values
        audio = np.squeeze(self.processor(audio,sampling_rate=16000).input_values)

        subject = "_".join(file_name.split("_")[:-1])
        one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)

    template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    speakers = set()
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                key = f.replace("wav", "npy")
                subject_id = "_".join(key.split("_")[:-1])
                speakers.add(subject_id)
                temp = templates[subject_id]
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if os.path.exists(vertice_path):
                    vertices = np.load(vertice_path,allow_pickle=True)
                    slice_len = args.max_seq_len
                    if(vertices.shape[0]%slice_len==0):
                        slice_count = int(vertices.shape[0]/slice_len)
                    else:
                        slice_count = int(vertices.shape[0]/slice_len) + 1
                    for idx in range(slice_count):
                        start = slice_len*idx
                        audio = speech_array[round(start*speech_array.shape[0]/vertices.shape[0]): round((start+slice_len)*speech_array.shape[0]/vertices.shape[0])]
                        vertice = vertices[start: start+slice_len, :]
                        data[key+f'_{idx}']["audio"] = audio
                        data[key+f'_{idx}']["name"] = f
                        data[key+f'_{idx}']["template"] = temp.reshape((-1)) 
                        data[key+f'_{idx}']["vertice"] = vertice

    subjects_dict = {}
    subjects_dict["train"] = list(speakers)

    for k, v in data.items():
        train_data.append(v)

    print(len(train_data))
    return train_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, subjects_dict = read_data(args)
    if(os.path.exists(os.path.join(args.save_path, 'subject_list.pkl'))):
        subjects_dict["train"] = pickle.load(open(os.path.join(args.save_path, 'subject_list.pkl'), 'rb'))
    train_data = Dataset(train_data, args.max_seq_len, subjects_dict, args, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    return dataset, subjects_dict["train"]

if __name__ == "__main__":
    get_dataloaders()
    
