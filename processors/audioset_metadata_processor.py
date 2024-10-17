import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

from audio_tokens_config import AudioTokensConfig
from set_seed import set_seed


class AudiosetMetadataProcessor:
    """
    AudiosetMetadataProcessor: Processes AudioSet metadata from JSON and CSV files.

    Loads ontology and segment data, creating mappings between labels, indices,
    names, and YouTube IDs. Provides methods to load and process metadata files.
    """

    def __init__(self, config: AudioTokensConfig):
        self.config = config
        set_seed(self.config.random_seed)
        self.logger = logging.getLogger()
        self.label_index: Dict[str, int] = {}
        self.index_label: Dict[int, str] = {}
        self.label_name: Dict[str, str] = {}
        self.ytid_labels: Dict[str, List[int]] = {}
        self._load_ontology()
        self._load_segment_data()

    def _load_ontology(self) -> None:
        ontology_path = Path(self.config.ontology_json_file)

        with ontology_path.open("r") as file:
            ontology_data = json.load(file)

        index = 0
        for item in ontology_data:
            if not item["restrictions"]:
                self.index_label[index] = item["id"]
                self.label_index[item["id"]] = index
                self.label_name[item["id"]] = item["name"]
                index += 1
        self.logger.info(f"Loaded {index} non-restricted classes")

    def _load_segment_data(self) -> None:
        """
        Load segment data from CSV files specified in the configuration.

        This method reads each CSV file, processes the rows, and populates
        the ytid_labels dictionary with YouTube IDs and their corresponding
        label indices.
        """
        for csv_file in self.config.csv_index_files:
            with open(csv_file, "r") as f:
                reader = csv.reader(f, skipinitialspace=True)

                # Skip header rows
                for _ in range(3):
                    next(reader)

                for row in reader:
                    ytid, label_str = row[0], row[3]
                    labels = label_str.split(",")
                    label_indexes = [
                        self.label_index[i] for i in labels if i in self.label_index
                    ]
                    self.ytid_labels[ytid] = label_indexes

            self.logger.info(
                f"Loaded segment data for {len(self.ytid_labels)} YouTube IDs"
            )
        no_labels = sum(1 for labels in self.ytid_labels.values() if not labels)
        self.logger.info(f"Number of YouTube IDs with no labels: {no_labels}")

        label_counts = [len(labels) for labels in self.ytid_labels.values()]
        avg_labels = sum(label_counts) / len(label_counts)
        self.logger.info(f"Average number of labels per YouTube ID: {avg_labels:.2f}")
        self.logger.info(f"Max number of labels for a YouTube ID: {max(label_counts)}")
        self.logger.info(f"Min number of labels for a YouTube ID: {min(label_counts)}")

    def get_all_ytids(self) -> List[str]:
        return list(self.ytid_labels.keys())

    def get_ytid_labels(self, ytid: str) -> List[int]:
        return self.ytid_labels.get(ytid, [])


if __name__ == "__main__":
    config = AudioTokensConfig()
    processor = AudiosetMetadataProcessor(config)
    index = 3
    ytid = processor.get_all_ytids()[index]
    label_indexes = processor.get_ytid_labels(ytid)
    print(ytid)
    print(label_indexes)
    labels = [processor.index_label[index] for index in label_indexes]
    print(labels)
    names = [processor.label_name[label] for label in labels]
    print(names)
