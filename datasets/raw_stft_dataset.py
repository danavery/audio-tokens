import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from audio_tokens_config import AudioTokensConfig
from processors.audioset_metadata_processor import AudiosetMetadataProcessor


class RawSTFTDataset(Dataset):
    def __init__(
        self,
        config: AudioTokensConfig,
        metadata_manager: AudiosetMetadataProcessor,
        split: str = "train",
    ):
        self.config = config
        self.metadata_manager = metadata_manager
        self.split = split

        self.ytids = self._load_split_data()
        self.stft_files = self._load_stft_files()

    def _load_split_data(self):
        with open(self.config.split_file, "r") as f:
            split_data = json.load(f)
        return split_data[self.split]

    def _load_stft_files(self):
        stft_files = []
        base_path = self.config.source_spec_path / self.split
        for ytid in self.ytids:
            file_path = os.path.join(base_path, f"{ytid}.npy")
            if os.path.exists(file_path):
                stft_files.append(file_path)
        return stft_files

    def __len__(self):
        return len(self.stft_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        filepath = self.stft_files[idx]
        stft = torch.tensor(np.load(filepath))
        stft = stft.transpose(0, 1)
        # Log the shape of the STFT
        # print(f"Spectrogram shape at index {idx}: {stft.shape}")
        ytid = os.path.splitext(os.path.basename(filepath))[0]
        label_indices = self.metadata_manager.ytid_labels[ytid]

        labels = torch.zeros(self.config.num_classes, dtype=torch.float)
        labels[label_indices] = 1.0

        return stft, {"labels": labels}  # stft, attention_mask, labels

    @staticmethod
    def collate_fn(batch):
        sequences, metadata = zip(*batch)
        labels = [item["labels"] for item in metadata]

        # Find the maximum number of time steps (second dimension) in this batch
        max_time_steps = max([seq.size(1) for seq in sequences])

        # Pad each sequence along the second dimension (time steps) to the maximum size in the batch
        padded_sequences = [F.pad(seq, (0, max_time_steps - seq.size(1)), "constant", 0) for seq in sequences]

        # Now pad the sequences along the first dimension (frequency bins)
        sequences = pad_sequence(padded_sequences, batch_first=True, padding_value=0).float()

        # Create attention masks based on the actual lengths of the sequences before padding
        attention_masks = [torch.ones(seq.size(1)) for seq in padded_sequences]
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0).float()

        # Stack the labels
        labels = torch.stack(labels).float()

        return sequences, {"attention_masks": attention_masks, "labels": labels}
