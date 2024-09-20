import logging
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from audio_tokens_config import AudioTokensConfig
from set_seed import set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ClusterCreator:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        set_seed(self.config.random_seed)
        self.gpu = faiss.get_num_gpus() > 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.config.use_convolution:
            self.conv = nn.Conv1d(
                in_channels=1,
                out_channels=self.config.num_kernels,
                kernel_size=self.config.kernel_size,
                padding=self.config.kernel_size // 2,
            ).to(self.device)

    def run(self):
        n_freq_bins = self.config.n_mels
        if self.config.use_convolution:
            n_freq_bins *= self.config.num_kernels

        self.logger.info("starting clustering")
        kmeans = faiss.Kmeans(
            n_freq_bins,
            self.config.vocab_size,
            niter=self.config.niter,
            verbose=True,
            gpu=self.gpu,
        )
        for i, batch in enumerate(
            self._batch_generator(self.config.clustering_batch_size)
        ):
            if i == 0:
                kmeans.train(batch)
            else:
                kmeans.train(batch, init_centroids=kmeans.centroids)

        centroids = kmeans.centroids
        self.logger.info(f"Centroids shape: {centroids.shape}")
        np.save(self.config.centroids_path, centroids)
        self.visualize_centroids(centroids)

    def apply_convolution(self, time_slice_batch):
        time_slice_batch = np.array(time_slice_batch)
        time_slice_batch = (
            torch.tensor(time_slice_batch, device=self.device).float().unsqueeze(1)
        )
        conv_output = self.conv(time_slice_batch)
        # Remove batch dimension and convert back to numpy
        return (
            conv_output.transpose(1, 2)
            .reshape(-1, self.config.num_kernels * self.config.n_mels)
            .cpu()
            .detach()
            .numpy()
        )

    def _batch_generator(self, batch_size):
        spec_dir = Path(self.config.source_path) / "train"
        batch = []
        for file in tqdm(spec_dir.glob("*.npy")):
            spec = np.load(file)
            spec = spec.T
            batch.extend(spec)

            while len(batch) >= batch_size:
                if self.config.use_convolution:
                    yield self.apply_convolution(batch[:batch_size])
                else:
                    yield np.array(batch[:batch_size], dtype=np.float32)
                batch = batch[batch_size:]

        if batch:
            if self.config.use_convolution:
                yield self.apply_convolution(batch)
            else:
                yield np.array(batch, dtype=np.float32)

    def visualize_centroids(self, centroids):
        pca = PCA(n_components=2)
        centroids_2d = pca.fit_transform(centroids)
        plt.figure(figsize=(10, 8))
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1])
        plt.title("2D PCA of Centroids")
        # plt.show()
        plt.savefig("output/centroids_visualization.png")
        plt.close()
        self.logger.info("Centroids visualization saved")

    def evaluate_clustering(self, data, labels):
        score = silhouette_score(data, labels, sample_size=10000)
        self.logger.info(f"Silhouette Score: {score}")


if __name__ == "__main__":
    set_seed()
    config = AudioTokensConfig()
    ClusterCreator(config).run()
