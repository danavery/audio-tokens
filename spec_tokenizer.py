import logging
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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

        for split, path in [
            ("train", self.config.train_spec_path),
            ("validation", self.config.val_spec_path),
        ]:
            tokenized_dir = self.dest_path / split
            tokenized_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Tokenizing {split} set: {path} --> {tokenized_dir}")
            all_tokens = self.tokenize(index, path, tokenized_dir)
            if split == "train":
                self.analyze_tokens(all_tokens)
                self.plot_token_distribution(all_tokens)

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

    def plot_token_distribution(self, all_tokens):
        # Count token frequencies
        token_counts = Counter(all_tokens)

        # Sort tokens by frequency
        sorted_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        tokens, frequencies = zip(*sorted_counts)

        # Create ranks (1 to number of unique tokens)
        ranks = range(1, len(tokens) + 1)

        plt.figure(figsize=(15, 10))

        # Plot full distribution
        plt.subplot(2, 1, 1)
        plt.plot(ranks, frequencies)
        plt.title("Distribution of Assigned Tokens (Sorted by Frequency)")
        plt.xlabel("Token Rank")
        plt.ylabel("Frequency")
        plt.yscale(
            "log"
        )  # Use log scale for y-axis to better visualize the distribution
        plt.xscale("log")  # Use log scale for x-axis to better visualize the long tail

        # Plot top 100 tokens
        plt.subplot(2, 1, 2)
        plt.bar(ranks, frequencies)
        plt.xlabel("Token Rank")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig("correct_token_distribution.png")
        plt.close()

        # Print some statistics
        total_tokens = sum(frequencies)
        cumulative_freq = np.cumsum(frequencies) / total_tokens
        top_80_percent = np.searchsorted(cumulative_freq, 0.8) + 1

        print(f"Total unique tokens: {len(tokens)}")
        print(f"Total token occurrences: {total_tokens}")
        print(
            f"Most common token (rank 1): Token {tokens[0]} (used {frequencies[0]} times)"
        )
        print(
            f"Least common token (rank {len(tokens)}): Token {tokens[-1]} (used {frequencies[-1]} times)"
        )
        print(f"Top {top_80_percent} tokens account for 80% of all token occurrences")
        print(
            f"Frequency ratio between most and least common: {frequencies[0] / frequencies[-1]:.2f}"
        )
        self.analyze_zipf_and_tail(frequencies)

    def analyze_zipf_and_tail(self, frequencies):
        ranks = np.arange(1, len(frequencies) + 1)
        log_ranks = np.log(ranks)
        log_frequencies = np.log(frequencies)

        # Fit a line to the log-log plot (excluding the first and last 10% for better fit)
        start_fit = int(0.1 * len(frequencies))
        end_fit = int(0.9 * len(frequencies))
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks[start_fit:end_fit], log_frequencies[start_fit:end_fit])

        # Plot the distribution and the fitted line
        plt.figure(figsize=(12, 8))
        plt.scatter(log_ranks, log_frequencies, alpha=0.5, label='Observed')
        plt.plot(log_ranks, intercept + slope * log_ranks, color='red', label=f'Fitted (slope = {slope:.2f})')
        plt.xlabel('Log Rank')
        plt.ylabel('Log Frequency')
        plt.title("Zipf's Law Analysis")
        plt.legend()
        plt.savefig('zipf_law_analysis.png')
        plt.close()

        # Analyze the tail
        total_occurrences = sum(frequencies)
        cumulative_freq = np.cumsum(frequencies) / total_occurrences
        tail_start = np.searchsorted(cumulative_freq, 0.8)  # Start of the last 20%
        tail_proportion = 1 - (tail_start / len(frequencies))

        print(f"Zipf's law slope: {slope:.2f} (closer to -1 indicates closer fit to Zipf's law)")
        print(f"R-squared value: {r_value**2:.2f}")
        print(f"Proportion of tokens in the tail (last 20% of occurrences): {tail_proportion:.2%}")
        print(f"Number of tokens accounting for 80% of occurrences: {tail_start}")


if __name__ == "__main__":
    spec_tokenizer_config = SpecTokenizerConfig()
    SpecTokenizer(spec_tokenizer_config).run()
