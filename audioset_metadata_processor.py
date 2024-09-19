import csv
import json
import logging
import random

from audio_tokens_config import AudioTokensConfig


class AudiosetMetadataProcessor:
    """
    AudiosetMetadataProcessor: Processes AudioSet metadata from JSON and CSV files.

    Loads ontology and segment data, creating mappings between labels, indices,
    names, and YouTube IDs. Provides methods to load and process metadata files.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        self.label_index = {}
        self.index_label = {}
        self.label_name = {}
        self.ytid_labels = {}
        self._load_ontology()
        self._load_segment_data()

    def _load_ontology(self):
        with open(self.config.ontology_json_file, "r") as file:
            ontology = json.load(file)
        for index, item in enumerate(ontology):
            self.index_label[index] = item["id"]
            self.label_index[item["id"]] = index
            self.label_name[item["id"]] = item["name"]

    def _load_segment_data(self):
        for csv_file in self.config.csv_index_files:
            with open(csv_file, "r") as f:
                reader = csv.reader(f, skipinitialspace=True)
                for _ in range(3):
                    next(reader)
                for row in reader:
                    ytid, label_str = row[0], row[3]
                    labels = label_str.split(",")
                    label_indexes = [self.label_index[i] for i in labels]
                    self.ytid_labels[ytid] = label_indexes

    def create_split_file(self):
        all_ytids = list(self.ytid_labels.keys())
        dataset_size = int(len(all_ytids) * self.config.dataset_ratio)
        random.Random(self.config.random_seed).shuffle(all_ytids)
        ytids = all_ytids[:dataset_size]
        split_index = int(len(ytids) * (1 - self.config.validation_ratio))
        split_data = {
            "train": ytids[:split_index],
            "validation": ytids[split_index:],
        }
        with open(self.config.split_file, "w") as f:
            json.dump(split_data, f)
        self.logger.info(f"Split file created at {self.config.split_file}")
        self.logger.info(f"Training: {len(split_data['train'])}")
        self.logger.info(f"Validation: {len(split_data['validation'])}")


if __name__ == "__main__":
    config = AudioTokensConfig()
    amp = AudiosetMetadataProcessor(config)
    amp.create_split_file()
