import csv
import json


class AudiosetMetadataProcessor:
    """
    AudiosetMetadataProcessor: Processes AudioSet metadata from JSON and CSV files.

    Loads ontology and segment data, creating mappings between labels, indices,
    names, and YouTube IDs. Provides methods to load and process metadata files.
    """

    def __init__(
        self,
        ontology_json="metadata/ontology.json",
        segment_csv="metadata/balanced_train_segments.csv",
    ):
        self.ontology_json = ontology_json
        self.segment_csv = segment_csv
        self.label_index = {}
        self.index_label = {}
        self.label_name = {}
        self.ytid_labels = {}
        self._create_dicts()

    def _create_dicts(self):
        ontology = self._load_ontology()
        self._make_ontology_dicts(ontology)
        self._make_ytid_label_dict(self.segment_csv)

    def _load_ontology(self):
        with open(self.ontology_json, "r") as file:
            return json.load(file)

    def _make_ontology_dicts(self, ontology):
        index = 0
        for item in ontology:
            # if item['restrictions']:
            #     continue
            self.index_label[index] = item["id"]
            self.label_index[item["id"]] = index
            self.label_name[item["id"]] = item["name"]
            index += 1

    def _make_ytid_label_dict(self, filename):
        with open(filename, "r") as f:
            reader = csv.reader(f, skipinitialspace=True)

            for _ in range(3):
                next(reader)

            for row in reader:
                ytid, label_str = row[0], row[3]
                labels = label_str.split(",")
                label_indexes = [self.label_index[i] for i in labels]
                self.ytid_labels[ytid] = label_indexes


if __name__ == "__main__":
    amp = AudiosetMetadataProcessor()
    print(len(amp.ytid_labels))
