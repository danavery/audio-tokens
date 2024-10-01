import logging
import shutil
from collections import Counter
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from tqdm import tqdm

from audio_tokens_config import AudioTokensConfig
from set_seed import set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SpecTokenizer:
    def __init__(self, config: AudioTokensConfig):
        self.config = config
        set_seed(self.config.random_seed)
        self.logger = logging.getLogger()
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.source_path = Path(self.config.source_spec_path)
        self.dest_tokenized_path = Path(self.config.dest_tokenized_path)
        self.centroid_path = Path(self.config.centroids_path)
        self.index = self.load_centroid_index()

        if self.config.use_convolution:
            self.conv = self.create_convolution_layer()

    def run(self):
        for split in ["train", "validation"]:
            source_spec_dir = self.source_path / split
            tokenized_dir = self.dest_tokenized_path / split

            self.setup_output_directory(tokenized_dir)
            self.logger.info(
                f"Tokenizing {split} set: {source_spec_dir} --> {tokenized_dir}"
            )
            all_tokens = self.tokenize_directory(source_spec_dir, tokenized_dir)
            if split == "train":
                self.analyze_tokens(all_tokens)
                self.plot_token_distribution(all_tokens)

    def tokenize_directory(self, source_dir: Path, tokenized_dir: Path):
        all_tokens = []
        spec_files = list(source_dir.glob("*.npy"))

        for i in tqdm(range(0, len(spec_files), self.config.tokenizer_batch_size)):
            batch_files = spec_files[i : i + self.config.tokenizer_batch_size]
            batch_tokens = self.process_batch(batch_files, tokenized_dir)
            all_tokens.extend(batch_tokens)

        return all_tokens

    def process_batch(self, batch_files, tokenized_dir: Path):
        batch_specs = [np.load(spec_file).T for spec_file in batch_files]
        batch_data = np.concatenate(batch_specs, axis=0)

        if self.config.use_convolution:
            processed_batch = self.apply_convolution(batch_data)
        else:
            processed_batch = batch_data.astype(np.float32)

        if processed_batch is not None and processed_batch.size > 0:
            processed_batch = self.normalize_vectors(processed_batch)
            _, tokens = self.index.search(processed_batch, 1)
            tokens = np.squeeze(tokens, 1)

            start = 0
            for spec_file, spec in zip(batch_files, batch_specs):
                end = start + len(spec)
                file_tokens = tokens[start:end]
                output_file = tokenized_dir / f"{spec_file.stem}.npy"
                np.save(output_file, file_tokens)
                start = end

            return tokens.tolist()

        return []

    def apply_convolution(self, batch):
        if len(batch) == 0:
            self.logger.warning("Received empty batch for convolution")
            return None
        batch_tensor = torch.tensor(batch, device=self.device).float().unsqueeze(1)
        conv_output = self.conv(batch_tensor)
        return (
            conv_output.transpose(1, 2)
            .reshape(-1, self.config.num_kernels * self.config.n_mels)
            .cpu()
            .detach()
            .numpy()
        )

    @staticmethod
    def normalize_vectors(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)

    def setup_output_directory(self, tokenized_dir):
        shutil.rmtree(tokenized_dir, ignore_errors=True)
        tokenized_dir.mkdir(parents=True)

    def create_convolution_layer(self):
        return nn.Conv1d(
            in_channels=1,
            out_channels=self.config.num_kernels,
            kernel_size=self.config.kernel_size,
            padding=self.config.kernel_size // 2,
        ).to(self.device)

    def load_centroid_index(self):
        centroids = np.load(self.centroid_path)
        index = faiss.IndexFlatL2(centroids.shape[1])
        index.add(centroids)
        return index

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
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_ranks[start_fit:end_fit], log_frequencies[start_fit:end_fit]
        )

        # Plot the distribution and the fitted line
        plt.figure(figsize=(12, 8))
        plt.scatter(log_ranks, log_frequencies, alpha=0.5, label="Observed")
        plt.plot(
            log_ranks,
            intercept + slope * log_ranks,
            color="red",
            label=f"Fitted (slope = {slope:.2f})",
        )
        plt.xlabel("Log Rank")
        plt.ylabel("Log Frequency")
        plt.title("Zipf's Law Analysis")
        plt.legend()
        plt.savefig("zipf_law_analysis.png")
        plt.close()

        # Analyze the tail
        total_occurrences = sum(frequencies)
        cumulative_freq = np.cumsum(frequencies) / total_occurrences
        tail_start = np.searchsorted(cumulative_freq, 0.8)  # Start of the last 20%
        tail_proportion = 1 - (tail_start / len(frequencies))

        print(
            f"Zipf's law slope: {slope:.2f} (closer to -1 indicates closer fit to Zipf's law)"
        )
        print(f"R-squared value: {r_value**2:.2f}")
        print(
            f"Proportion of tokens in the tail (last 20% of occurrences): {tail_proportion:.2%}"
        )
        print(f"Number of tokens accounting for 80% of occurrences: {tail_start}")


if __name__ == "__main__":
    spec_tokenizer_config = AudioTokensConfig()
    SpecTokenizer(spec_tokenizer_config).run()
