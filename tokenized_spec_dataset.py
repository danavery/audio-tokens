import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenizedSpecDataset(Dataset):
    def __init__(self, config, data_manager, split="train"):
        self.config = config
        self.data_manager = data_manager
        self.split = split
        self.tokenized_spec_files = []
        self.ytids = []
        self._load_split_data()
        self._load_tokenized_specs()

    def _load_split_data(self):
        with open(self.config.split_file, "r") as f:
            split_data = json.load(f)
        self.ytids = split_data[self.split]

    def _load_tokenized_specs(self):
        base_path = (
            self.config.tokenized_train_dir if self.split == "train" else self.config.tokenized_val_dir
        )
        for ytid in self.ytids:
            file_path = os.path.join(base_path, f"{ytid}.npy")
            if os.path.exists(file_path):
                self.tokenized_spec_files.append(file_path)
            else:
                pass
                # print(f"Warning: File not found for YouTube ID {ytid}")

    def __len__(self):
        return len(self.tokenized_spec_files)

    def __getitem__(self, idx):
        filepath = self.tokenized_spec_files[idx]
        seq = torch.tensor(np.load(filepath))
        attention_mask = torch.ones_like(seq)

        ytid = os.path.splitext(os.path.basename(filepath))[0]
        label_indices = self.data_manager.ytid_labels[ytid]

        label_tensor = torch.zeros(self.config.num_classes, dtype=torch.float)
        label_tensor[label_indices] = 1.0

        return seq, attention_mask, label_tensor
