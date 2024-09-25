import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from audio_tokens_config import AudioTokensConfig
from audioset_metadata_processor import AudiosetMetadataProcessor
from tokenized_spec_dataset import TokenizedSpecDataset


class DatasetTester:
    def __init__(self, config):
        self.config = config
        self.metadata_manager = AudiosetMetadataProcessor(self.config)

    def run(self):
        train_loader, val_loader = self._initialize_data_loaders()
        self._get_data_loader_counts(train_loader, val_loader)
        self._check_for_overlap(train_loader, val_loader)
        self._get_sequence_samples(train_loader, val_loader)
        self._get_label_samples(train_loader, val_loader)

    def _get_sequence_samples(self, train_loader, val_loader):

        torch.set_printoptions(threshold=float('inf'))
        # Define the number of samples you want
        num_samples = 5
        train_remainder_samples = len(train_loader.dataset) - num_samples

        sample_train_examples, _ = random_split(train_loader.dataset, [num_samples, train_remainder_samples])

        for i, sample in enumerate(sample_train_examples):
            print(f"Sample length: {sample[0].shape}")
            print(f"Attention mask length: {torch.sum(sample[1])}")
            print(f"Sample {i+1}: {sample}")

        val_remainder_samples = len(val_loader.dataset) - num_samples

        sample_val_examples, _ = random_split(val_loader.dataset, [num_samples, val_remainder_samples])

        for i, sample in enumerate(sample_val_examples):
            print(f"Sample length: {sample[0].shape}")
            print(f"Attention mask length: {torch.sum(sample[1])}")
            print(f"Sample {i+1}: {sample}")

    def _get_label_samples(self, train_loader, val_loader):
        train_ytids = list(self._extract_ytids(train_loader))
        val_ytids = list(self._extract_ytids(val_loader))

        random.shuffle(train_ytids)
        random.shuffle(val_ytids)

        print("sample train labels")
        for ytid in train_ytids[:10]:
            print(f"{ytid} Labels: {self.metadata_manager.ytid_labels[ytid]} ")
        print("sample val labels")
        for ytid in val_ytids[:10]:
            print(f"{ytid} Labels: {self.metadata_manager.ytid_labels[ytid]} ")

    def _check_for_overlap(self, train_loader, val_loader):
        train_ytids = self._extract_ytids(train_loader)
        val_ytids = self._extract_ytids(val_loader)
        overlap = train_ytids.intersection(val_ytids)
        if overlap:
            print(f"{len(overlap)} items found in both sets!")
        else:
            print("No overlap in the sets")

    def _get_data_loader_counts(self, train_loader, val_loader):
        print(f"Train dataloader size: {len(train_loader.dataset)}")
        print(f"Val dataloader size: {len(val_loader.dataset)}")

    def _extract_ytids(self, dataloader):
        ytids = set()
        for i in range(len(dataloader.dataset)):
            # Access the dataset directly and get the YouTube ID from the file path
            filepath = dataloader.dataset.tokenized_spec_files[i]
            ytid = os.path.splitext(os.path.basename(filepath))[0]
            ytids.add(ytid)
        return ytids

    def _initialize_data_loaders(self):
        train_dataset = TokenizedSpecDataset(
            self.config,
            self.metadata_manager,
            split="train",
        )
        val_dataset = TokenizedSpecDataset(
            self.config, self.metadata_manager, split="validation"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader, val_loader

    def collate_fn(self, batch):
        input_ids = [item[0] for item in batch]
        attention_masks = [item[1] for item in batch]
        labels = [item[2] for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).long()
        attention_masks = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        ).float()
        labels = torch.stack(labels).float()
        return input_ids, attention_masks, labels


if __name__ == "__main__":
    config = AudioTokensConfig()
    tester = DatasetTester(config)
    tester.run()
