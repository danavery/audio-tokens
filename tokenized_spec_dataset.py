import os

import numpy as np
import torch
from torch.utils.data import Dataset

from audioset_metadata_processor import AudiosetMetadataProcessor


class TokenizedSpecDataset(Dataset):
    def __init__(self, tokenized_spec_files, label_fn=None, num_classes=527):
        self.tokenized_spec_files = tokenized_spec_files
        # print(self.tokenized_spec_files)
        self.label_fn = label_fn or self.default_get_label
        self.ytid_labels = AudiosetMetadataProcessor().ytid_labels
        self.num_classes = num_classes

    def __len__(self):
        return len(self.tokenized_spec_files)

    def __getitem__(self, idx):
        filepath = self.tokenized_spec_files[idx]
        seq = torch.tensor(np.load(filepath))
        attention_mask = torch.ones_like(seq)
        label_indices = self.label_fn(filepath)

        label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
        label_tensor[label_indices] = 1.0
        # print(f"Sample {idx} label sum: {label_tensor.sum().item()}")
        return seq, attention_mask, label_tensor

    def default_get_label(self, file_path):
        ytid = os.path.splitext(os.path.basename(file_path))[0]
        return self.ytid_labels[ytid]
