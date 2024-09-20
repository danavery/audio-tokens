import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class AudioTokensConfig:
    # AudiosetMetadataProcessor
    csv_index_files = ["metadata/balanced_train_segments.csv", "metadata/unbalanced_train_segments.csv"]
    ontology_json_file = "metadata/ontology.json"
    dataset_ratio = 0.02  # portion of all ytids to use
    validation_ratio = 0.1  # portion of dataset to use as validation set
    random_seed = 42

    # AudiosetMetadataProcessor and SpectrogramProcessor
    split_file: str = "output/bal_unbal_train_val_data_split.json"

    # SpectrogramProcessor
    source_ytids: List[str] = field(default_factory=list)
    source_parent: str = "/media/davery/audioset"
    source_sets: List[str] = field(default_factory=lambda: ["unbal_train", "bal_train"])
    dest_spec_path: str = "processed/"
    common_sr: int = 22050
    normalize: bool = True
    n_mels: int = 64
    n_fft: int = 256
    hop_length: int = 128
    n_segments: int = 0
    spectrogram_batch_size: int = 10000

    # ClusterCreator and ModelTrainer
    vocab_size: int = 500

    # ClusterCreator and SpecTokenizer
    train_spec_path: Path = Path("processed/train_specs.pkl")

    # ClusterCreator
    niter: int = 20
    centroids_path: Path = Path("output/centroids.npy")
    use_convolution: bool = False
    num_kernels: int = 8
    kernel_size: int = 3
    clustering_batch_size: int = 1000000

    # SpecTokenizer config
    source_path: str = "processed/"
    dest_tokenized_path: str = "tokenized/"
    centroid_path: str = "output/centroids.npy"
    train_spec_path: str = "processed/train_specs.pkl"
    val_spec_path: str = "processed/validation_specs.pkl"
    tokenizer_batch_size: int = 10000

    # ModelTrainer
    use_wandb: bool = True
    seq_dir: str = "tokenized/"
    train_dir: str = "tokenized/train/"
    val_dir: str = "tokenized/validation/"
    model_type: str = "lstm"
    num_layers: int = 1
    epochs: int = 200
    hidden_size: int = 768
    batch_size: int = 128
    num_workers: int = 8
    learning_rate: float = 1e-3
    num_classes: int = 631
    prediction_threshold: float = 0.2
    lstm_embed_dim: int = 128
    lstm_hidden_dim: int = 256
    dropout: float = 0.5
