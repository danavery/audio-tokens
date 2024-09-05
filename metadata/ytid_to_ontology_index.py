import csv
from ontology_id_to_index import get_id_index_dict, load_ontology


def parse_csv(filename, id_to_index_dict):
    ytid_labels = {}
    with open(filename, "r") as f:
        # Create a CSV reader object
        reader = csv.reader(f, skipinitialspace=True)

        # Skip the first 3 comment lines
        for _ in range(3):
            next(reader)

        # Process the remaining lines
        for row in reader:
            ytid, label_str = row[0], row[3]
            labels = label_str.split(",")
            label_indexes = [id_to_index_dict[i] for i in labels]
            ytid_labels[ytid] = label_indexes
    return ytid_labels


if __name__ == "__main__":
    ontology = load_ontology("ontology.json")
    id_index, _ = get_id_index_dict(ontology)
    ytid_labels = parse_csv("balanced_train_segments.csv", id_index)
    # print(ytid_labels)
