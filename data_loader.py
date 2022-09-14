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
    def __init__(self, data,subjects_dict,data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        key = self.data[index]["name"]
        audio = self.data[index]["audio"]
        exp_jaw = self.data[index]["exp_jaw"]
        random_len = random.randrange(30 * 2, min(30 * 10, exp_jaw.shape[0] - 30*2))
        start = random.randrange(0, exp_jaw.shape[0] - random_len)
        audio = audio[round(start*audio.shape[0]/exp_jaw.shape[0]): round((start+random_len)*audio.shape[0]/exp_jaw.shape[0])]
        exp_jaw = exp_jaw[start: start+random_len, :]

        one_hot = self.one_hot_labels[self.subjects_dict["train"].index(key.split('_')[0])]
        return torch.FloatTensor(audio),torch.FloatTensor(exp_jaw), torch.FloatTensor(one_hot), key

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []

    root_path = args.dataset
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    speakers = os.listdir(root_path)
    for speaker in speakers:
        sentences = os.listdir(os.path.join(root_path, speaker))
        for sentence in sentences:
            wav_path = os.path.join(root_path, speaker, sentence, 'audio.wav')
            speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
            input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
            key = speaker + '_' + sentence
            data[key]["audio"] = input_values
            data[key]["name"] = key
            exp_jaw_path = os.path.join(root_path, speaker, sentence, 'exp_jaw.npy')
            data[key]["exp_jaw"] = np.load(exp_jaw_path, allow_pickle=True)

    subjects_dict = {}
    subjects_dict["train"] = speakers

    for k, v in data.items():
        train_data.append(v)

    print(len(train_data))
    return train_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    return dataset, len(dataset["train"])

if __name__ == "__main__":
    get_dataloaders()
    
