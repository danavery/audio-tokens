import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class AudioTokensConfig:
    random_seed: int = 4242

    # AudiosetMetadataProcessor
    csv_index_files: List[str] = field(
        default_factory=lambda: [
            "metadata/balanced_train_segments.csv",
            # "metadata/unbalanced_train_segments.csv",
        ]
    )
    ontology_json_file: str = "metadata/ontology.json"
    dataset_ratio: float = 1  # portion of all ytids to use
    validation_ratio: float = 0.1  # portion of dataset to use as validation set

    # AudiosetMetadataProcessor and SpectrogramProcessor
    split_file: str = "output/bal_train_val_data_split.json"

    # SpectrogramProcessor
    audio_source_path: str = "/media/davery/audioset"
    audio_source_sets: List[str] = field(default_factory=lambda: ["bal_train"])
    dest_spec_path: str = "spectrograms/"
    common_sr: int = 22050
    normalize: bool = True
    n_mels: int = 64
    n_fft: int = 4096
    hop_length: int = 2048
    n_segments: int = 0
    spectrogram_batch_size: int = 10000

    # ClusterCreator and ModelTrainer
    vocab_size: int = 500

    # ClusterCreator
    niter: int = 20
    use_convolution: bool = True
    num_kernels: int = 10
    kernel_size: int = 3
    clustering_batch_size: int = 10000

    # ClusterCreator and SpecTokenizer
    centroids_path: Path = Path("output/centroids.npy")
    source_spec_path: str = "spectrograms/"

    # SpecTokenizer config
    dest_tokenized_path: str = "tokenized_audio/"
    tokenizer_batch_size: int = 10000

    # ModelTrainer
    use_wandb: bool = True
    tokenized_train_dir: str = "tokenized_audio/train/"
    tokenized_val_dir: str = "tokenized_audio/validation/"
    model_type: str = "lstm"
    num_layers: int = 1
    epochs: int = 5000
    hidden_size: int = 768
    training_batch_size: int = 256
    num_workers: int = 8
    learning_rate: float = 1e-5
    num_classes: int = 543
    prediction_threshold: float = 0.2
    lstm_embed_dim: int = 128
    lstm_hidden_dim: int = 256
    dropout: float = 0.5
