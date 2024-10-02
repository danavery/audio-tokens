import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from audioset_metadata_processor import AudiosetMetadataProcessor
from audio_tokens_config import AudioTokensConfig


class TokenizedSpecDataset(Dataset):
    def __init__(
        self,
        config: AudioTokensConfig,
        data_manager: AudiosetMetadataProcessor,
        split: str = "train",
    ):
        self.config = config
        self.data_manager = data_manager
        self.split = split

        self.ytids = self._load_split_data()
        self.tokenized_spec_files = self._load_tokenized_specs()

    def _load_split_data(self):
        with open(self.config.split_file, "r") as f:
            split_data = json.load(f)
        return split_data[self.split]

    def _load_tokenized_specs(self) -> None:
        tokenized_spec_files = []
        base_path = (
            self.config.tokenized_train_dir
            if self.split == "train"
            else self.config.tokenized_val_dir
        )
        for ytid in self.ytids:
            file_path = os.path.join(base_path, f"{ytid}.npy")
            if os.path.exists(file_path):
                tokenized_spec_files.append(file_path)
            else:
                pass  # not all of the original AudioSet examples still exist on YT, so they're not in this dataset
                # print(f"Warning: File not found for YouTube ID {ytid}")
        return tokenized_spec_files

    def __len__(self):
        return len(self.tokenized_spec_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        filepath = self.tokenized_spec_files[idx]
        seq = torch.tensor(np.load(filepath))

        ytid = os.path.splitext(os.path.basename(filepath))[0]
        label_indices = self.data_manager.ytid_labels[ytid]

        label_tensor = torch.zeros(self.config.num_classes, dtype=torch.float)
        label_tensor[label_indices] = 1.0

        return seq, {"labels": label_tensor}
