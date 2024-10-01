import json
import logging
import random
from pathlib import Path
from typing import List

from audio_tokens_config import AudioTokensConfig
from audioset_metadata_processor import AudiosetMetadataProcessor


class DatasetSplitter:
    def __init__(self, config: AudioTokensConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_split_file(self, metadata_processor: AudiosetMetadataProcessor):
        all_ytids = metadata_processor.get_all_ytids()
        train_ytids, val_ytids = self._split_data(all_ytids)

        split_data = {
            "train": train_ytids,
            "validation": val_ytids,
        }

        split_file_path = Path(self.config.split_file)
        split_file_path.parent.mkdir(parents=True, exist_ok=True)

        with split_file_path.open("w") as f:
            json.dump(split_data, f)

        self.logger.info(f"Split file created at {split_file_path}")
        self.logger.info(f"Training: {len(split_data['train'])}")
        self.logger.info(f"Validation: {len(split_data['validation'])}")

    def _split_data(self, ytids: List[str]):
        random.seed(self.config.random_seed)
        random.shuffle(ytids)

        dataset_size = int(len(ytids) * self.config.dataset_ratio)
        ytids = ytids[:dataset_size]
        split_index = int(len(ytids) * (1 - self.config.validation_ratio))
        return ytids[:split_index], ytids[split_index:]


if __name__ == "__main__":
    config = AudioTokensConfig()
    metadata_processor = AudiosetMetadataProcessor(config)
    splitter = DatasetSplitter(config)
    splitter.create_split_file(metadata_processor)
