import logging
import pickle

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from audio_tokens_config import AudioTokensConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ClusterCreator:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.gpu = faiss.get_num_gpus() > 0
        if self.config.use_convolution:
            self.conv = nn.Conv1D(
                in_channels=1,
                out_channels=self.config.num_kernels,
                kernel_size=self.config.kernel_size,
                padding=self.config.kernel_size // 2,
            )

    def run(self):
        time_slices_array = self._make_time_slices_array()
        self.logger.info(f"{time_slices_array.shape=}")
        n_freq_bins = time_slices_array.shape[1]

        # Perform clustering
        self.logger.info("starting clustering")
        kmeans = faiss.Kmeans(
            n_freq_bins,
            self.config.vocab_size,
            niter=self.config.niter,
            verbose=True,
            gpu=self.gpu,
        )
        kmeans.train(time_slices_array)

        # Get the centroids
        centroids = kmeans.centroids
        self.logger.info(centroids.shape)
        np.save(self.config.centroids_path, centroids)
        self.visualize_centroids(centroids)

    def apply_convolution(self, time_slice):
        # Ensure time_slice is a PyTorch tensor and add batch and channel dimensions
        time_slice = torch.tensor(time_slice).float().unsqueeze(0).unsqueeze(0)
        # Apply convolution
        conv_output = self.conv(time_slice)
        # Remove batch dimension and convert back to numpy
        return conv_output.squeeze(0).detach().numpy()

    def _make_time_slices_array(self):
        time_slices = []
        with open(self.config.train_spec_path, "rb") as f:
            specs = pickle.load(f)
        for spec_record in tqdm(specs):
            self.logger.debug(f"{spec_record['filename']}:")
            spec = spec_record["spec"].T
            self.logger.debug(spec.shape)
            if spec.ndim != 2:
                raise ValueError(f"Invalid spectrogram shape: {spec.shape}")

            if self.config.use_convolution:
                for time_slice in spec:
                    conv_output = self.apply_convolution(time_slice)
                    time_slices.append(conv_output.flatten())
            else:
                time_slices.extend(spec)
        self.logger.info(len(time_slices))
        return np.array(time_slices)

    def visualize_centroids(self, centroids):
        pca = PCA(n_components=2)
        centroids_2d = pca.fit_transform(centroids)
        plt.figure(figsize=(10, 8))
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1])
        plt.title("2D PCA of Centroids")
        plt.show()
        plt.savefig("output/centroids_visualization.png")
        plt.close()
        self.logger.info("Centroids visualization saved")

    def evaluate_clustering(self, data, labels):
        score = silhouette_score(data, labels, sample_size=10000)
        self.logger.info(f"Silhouette Score: {score}")


if __name__ == "__main__":
    config = AudioTokensConfig()
    ClusterCreator(config).run()
