import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class ClusterCreatorConfig:
    n_clusters: int = 500
    train_spec_path: Path = Path("processed/train_specs.pkl")
    niter: int = 20
    centroids_path = Path("output/centroids.npy")


class ClusterCreator:
    def __init__(self, config):
        self.config = config
        self.gpu = faiss.get_num_gpus() > 0
        self.logger = logging.getLogger(__name__)

    def run(self):
        time_slices_array = self.make_time_slices_array()
        self.logger.info(f"{time_slices_array.shape=}")
        n_freq_bins = time_slices_array.shape[1]

        # Perform clustering
        kmeans = faiss.Kmeans(
            n_freq_bins,
            self.config.n_clusters,
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

    def make_time_slices_array(self):
        time_slices = []
        with open(self.config.train_spec_path, "rb") as f:
            specs = pickle.load(f)
        for spec_record in specs:
            self.logger.debug(f"{spec_record['filename']}:")
            spec = spec_record["spec"].T
            self.logger.debug(spec.shape)
            if spec.ndim != 2:
                raise ValueError(f"Invalid spectrogram shape: {spec.shape}")
            time_slices.append(spec)
        self.logger.info(len(time_slices))

        return np.vstack(time_slices)

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
    cluster_creator_config = ClusterCreatorConfig(n_clusters=50, niter=100)
    ClusterCreator(cluster_creator_config).run()
