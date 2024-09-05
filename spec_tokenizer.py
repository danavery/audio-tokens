import logging
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class SpecTokenizerConfig:
    source_path: str = "processed/"
    dest_path: str = "tokenized/"
    centroid_path: str = "output/centroids.npy"
    train_spec_path: str = "processed/train_specs.pkl"
    val_spec_path: str = "processed/validation_specs.pkl"


class SpecTokenizer:
    def __init__(self, config: SpecTokenizerConfig):
        self.config = config
        self.source_path = Path(self.config.source_path)
        self.dest_path = Path(self.config.dest_path)
        self.centroid_path = Path(self.config.centroid_path)
        self.logger = logging.getLogger()

    def run(self):
        index = self.get_centroid_index()

        for split, path in [('train', self.config.train_spec_path), ('validation', self.config.val_spec_path)]:
            tokenized_dir = self.dest_path / split
            tokenized_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Tokenizing {split} set: {path} --> {tokenized_dir}")
            all_tokens = self.tokenize(index, path, tokenized_dir)
            self.analyze_tokens(all_tokens)

    def get_centroid_index(self):
        # Load the centroids
        centroids = np.load(self.centroid_path)

        # Create a FAISS index and add the centroids
        d = centroids.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(centroids)
        return index

    def tokenize(self, index, spec_pkl_path, tokenized_dir):
        all_tokens = []
        with open(spec_pkl_path, "rb") as f:
            spec_records = pickle.load(f)
        for spec_record in tqdm(spec_records):
            spec, filename = spec_record["spec"], spec_record["filename"]
            spec = spec.T
            _, tokens = index.search(spec, 1)
            tokens = np.squeeze(tokens, 1)

            if tokens.size != spec.shape[0]:
                raise ValueError(f"Token count mismatch for {filename}")
            all_tokens.extend(tokens)
            tokenized_path = tokenized_dir / Path(filename).stem
            np.save(tokenized_path, tokens)
        return all_tokens

    def analyze_tokens(self, all_tokens):
        token_counts = Counter(all_tokens)

        plt.figure(figsize=(12, 6))
        plt.bar(token_counts.keys(), token_counts.values())
        plt.title("Distribution of Assigned Tokens")
        plt.xlabel("Token ID")
        plt.ylabel("Frequency")
        plt.show()
        plt.savefig("output/token_distribution.png")
        plt.close()

        self.logger.info(f"Total tokens: {len(all_tokens)}")
        self.logger.info(f"Unique tokens: {len(token_counts)}")
        self.logger.info(f"Most common token: {token_counts.most_common(1)}")
        self.logger.info(f"Least common token: {token_counts.most_common()[-1]}")


if __name__ == "__main__":
    spec_tokenizer_config = SpecTokenizerConfig()
    SpecTokenizer(spec_tokenizer_config).run()
