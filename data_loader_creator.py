from typing import Tuple

from torch.utils.data import DataLoader

from audio_tokens_config import AudioTokensConfig
from audioset_metadata_processor import AudiosetMetadataProcessor
from raw_stft_dataset import RawSTFTDataset
from raw_stft_flat_dataset import RawSTFTFlatDataset
from tokenized_spec_dataset import TokenizedSpecDataset


class DataLoaderCreator:
    def __init__(self, config: AudioTokensConfig):
        self.config = config

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        dataset_class = self._get_dataset_class()
        train_dataset, val_dataset = self._get_datasets()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=dataset_class.collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=dataset_class.collate_fn,
        )
        return train_loader, val_loader

    def _get_dataset_class(self):
        if self.config.dataset_type == "TokenizedSpecDataset":
            return TokenizedSpecDataset
        elif self.config.dataset_type == "RawSTFTDataset":
            return RawSTFTDataset
        elif self.config.dataset_type == "RawSTFTFlatDataset":
            return RawSTFTFlatDataset
        else:
            raise ValueError(f"Unsupported dataset type: {self.config.dataset_type}")

    def _get_datasets(self):
        dataset_type = self._get_dataset_class()
        metadata_manager = AudiosetMetadataProcessor(self.config)
        self.train_dataset = dataset_type(self.config, metadata_manager, split="train")
        self.val_dataset = dataset_type(
            self.config, metadata_manager, split="validation"
        )
        return self.train_dataset, self.val_dataset
