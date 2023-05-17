import glob
import sys
import json
import numpy as np
import os
import torch
import torchaudio
from dotenv import load_dotenv
from DataTransformation import DataTransformation

cur_dir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(cur_dir, "..", ".env"))

NUM_SAMPLES = int(os.getenv("NUM_SAMPLES"))
N_FFT = int(os.getenv("N_FFT"))  # Parameter for spectrogram
HOP_LENGTH = int(os.getenv("HOP_LENGTH"))  # Parameter for spectrogram
METADATA_FILENAME = "metadata.json"


class AcousticDataset(torch.utils.data.Dataset):
    """
    Class handling usage of the audio dataset.
    """

    def __init__(self, dataset_path, enable_doa = False):
        """
        Constructs all the necessary attributes for the AcousticDataset class.
        Args:
            dataset_path (string): The directory on the computer to the dataset to be loaded.
        """
        self.data_modification = DataTransformation(n_fft = N_FFT,hop_length=HOP_LENGTH, sample_rate = 16000, audio_length_in_seconds= NUM_SAMPLES/16000)  # hardcode sample rate
        self.data = glob.glob(os.path.join(dataset_path, "*", "*.wav"), recursive=True)
        self.intToName = {}
        self.nameToInt = {}
        self.class_weight = []
        self.enable_doa = enable_doa
        labels = []

        if enable_doa:
            with open(os.path.join(dataset_path, METADATA_FILENAME), "r") as f:
                self.metadata = json.load(f)

        dirList = os.listdir(dataset_path)
        for item in dirList:  # Search the dataset folder for labels(subfolder)
            if item not in labels and os.path.isdir(os.path.join(dataset_path, item)):
                labels.append(item)

        labels.sort()

        file_counter = 0
        for (label, target_counter) in zip(labels, range(len(labels))):
            nb_file = len(os.listdir(os.path.join(dataset_path, label)))
            self.class_weight.append(nb_file)
            file_counter += nb_file
            self.intToName[target_counter] = label
            self.nameToInt[label] = target_counter

        # Computing weight for each classes
        for i in range(len(self.class_weight)):
            self.class_weight[i] = 1 / (self.class_weight[i] / file_counter)

        self.class_weight = torch.Tensor(self.class_weight) / torch.sum(torch.Tensor(self.class_weight))

        with (open(os.path.join(cur_dir, "LABELS.json"), "w", encoding="utf-8")) as labelFile:
            json.dump(self.intToName, labelFile, indent=4, ensure_ascii=False)

    def __len__(self):
        """
        This specifies the number of items from the dataset to be chosen.
        Returns:
            Length (int) of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        This method is called when an item is called. This returns the input tensor for the neural network, the
        category label of the .wav file and the doa of the person speaking.
        Args:
            idx (int): The index of the .wav file required.

        Returns:
            input_rnn (tensor): The 5 channels input tensor to the neural network.
            category_idx (int): The category index of the .wav file.
            doa (tensor): The doa of the person speaking. These values represent the unitary x, y and z directions.
        """
        item_path = self.data[idx]

        input_rnn = self.data_modification.from_audio_file(item_path)

        # get category and doa from metadata
        names = item_path.split(os.path.sep)[-2:]
        filename = os.path.join(names[0], names[1])
        category = item_path.split(os.path.sep)[-2]
        category_idx = self.nameToInt[category]
        if self.enable_doa:
            doa = [float(item) for item in self.metadata[filename]["sample_position"].values()]
        else:
            doa = [0, 0, 0]
        
        return input_rnn, category_idx, torch.Tensor(doa)
