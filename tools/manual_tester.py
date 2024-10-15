from pathlib import Path
import logging

import faiss
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Resample

from audio_tokens_config import AudioTokensConfig
from audioset_metadata_processor import AudiosetMetadataProcessor
from simple_lstm_token_classifier import SimpleLSTMTokenClassifier


class ManualTester:
    def __init__(self, config, model_file):
        self.config = config
        self.logger = logging.getLogger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.get_lstm_model()
        self.model.load_state_dict(torch.load(model_file, weights_only=False))
        self.spec_transformer = MelSpectrogram(
            sample_rate=self.config.common_sr,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        self.amplitude_to_db_transformer = AmplitudeToDB()
        self.metadata_manager = AudiosetMetadataProcessor(self.config)
        self.n_results = 30

    def run(self, example_ytid):
        example_ytid = example_ytid.replace(".npy", "")
        spec = self.get_spectrogram(example_ytid)
        spec = spec.T

        self.logger.info(spec.shape)
        seq = self.get_sequence(spec)
        self.logger.info(seq.shape)
        self.logger.info(seq)
        labels = self.get_labels(example_ytid)
        self.logger.info(labels)
        ontology_labels = self.get_ontology_labels(labels)
        # self.logger.info(ontology_labels)
        label_names = self.get_label_names(ontology_labels)
        self.logger.info(f"Labels: {label_names}")
        top_N_indices, top_N_values = self.get_predictions(seq)
        self.output_results(top_N_indices, top_N_values, label_names)

    def output_results(self, indices, values, index_names):
        self.logger.info("Rank\tCategory\tValue")
        for i, (index, value) in enumerate(zip(indices, values)):
            index_name = self.metadata_manager.label_name[self.metadata_manager.index_label[index.item()]]
            self.logger.info(f"{i+1}\t{value:.3f}\t{index_name} {'*' if index_name in index_names else ''}")

    def get_predictions(self, seq):
        seq = torch.tensor(np.array([seq])).to(self.device)
        print(seq.shape)
        seq = seq[:, :512]
        outputs = self.model(seq, {"attention_masks": torch.ones_like(seq).to(self.device)})
        outputs = torch.sigmoid(outputs)
        sorted_values, sorted_indices = torch.sort(outputs, descending=True)  # Get top 10 predictions
        top_N_values = sorted_values[0, :self.n_results]
        top_N_indices = sorted_indices[0, :self.n_results]
        return top_N_indices, top_N_values

    def get_labels(self, ytid):
        return self.metadata_manager.ytid_labels[ytid]

    def get_ontology_labels(self, labels):
        return [self.metadata_manager.index_label[label] for label in labels]

    def get_label_names(self, ontology_labels):
        return [self.metadata_manager.label_name[label] for label in ontology_labels]

    def get_sequence(self, spec):
        index = self.get_centroid_index()
        _, tokens = index.search(spec, 1)
        tokens = np.squeeze(tokens, 1)
        return tokens

    def get_centroid_index(self):
        # Load the centroids
        centroids = np.load(self.config.centroids_path)
        d = centroids.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(centroids)
        return index

    def generate_mel_spectrogram(self, audio):
        mel_spec = self.spec_transformer(audio).squeeze(0)
        mel_spec_db = self.amplitude_to_db_transformer(mel_spec)
        return mel_spec_db

    def get_spectrogram(self, ytid):
        ytid = ytid.replace(".flac", "")
        print(f"looking for {ytid}")
        try:
            for source_set in self.config.audio_source_sets:
                audio_file_path = Path(
                    f"{self.config.audio_source_path}/{source_set}/{ytid[:2]}/{ytid}.flac"
                )
                self.logger.info(audio_file_path)
                if audio_file_path.exists():
                    found = True
                    break
                else:
                    found = False
            if found:
                self.logger.info(audio_file_path)
                waveform = self.preprocess_waveform(audio_file_path)
            else:
                self.logger.debug(f"Not found: {audio_file_path}")
        except RuntimeError as e:
            self.logger.debug(f"{e}: {audio_file_path}")
        spec = self.generate_mel_spectrogram(waveform)
        if self.config.normalize:
            spec = self.normalize_spectrogram(spec)
        if self.check_for_nan_inf(spec, f"{ytid}"):
            self.logger.debug(f"Bad file: {audio_file_path}")
        return spec

    @staticmethod
    def normalize_spectrogram(spec):
        spectrogram = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec))
        return spectrogram

    def check_for_nan_inf(self, data, name="data"):
        if torch.isnan(data).any():
            self.logger.debug(f"Warning: NaN values found in {name}")
            self.logger.debug(
                f"Indices of NaN values: {torch.nonzero(torch.isnan(data), as_tuple=True)}"
            )
            return True
        if torch.isinf(data).any():
            self.logger.debug(f"Warning: Inf values found in {name}")
            self.logger.debug(
                f"Indices of NaN values: {torch.nonzero(torch.isinf(data), as_tuple=True)}"
            )
            return True
        return False

    def preprocess_waveform(self, audio_file_path):
        waveform, sr = torchaudio.load(audio_file_path)
        waveform = self.convert_to_mono(waveform)
        waveform = self.resample(waveform, sr)
        return waveform

    @staticmethod
    def convert_to_mono(waveform):
        if waveform.shape[0] > 1:  # Check if the audio is stereo or surround
            mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
            return mono_waveform
        else:
            return waveform  # If the audio is already mono, return it as is

    def resample(self, waveform, sr):
        if sr == self.config.common_sr:
            return waveform
        resampler = Resample(sr, self.config.common_sr)
        waveform = resampler(waveform)
        return waveform

    def get_lstm_model(self):
        return SimpleLSTMTokenClassifier(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.lstm_embed_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.num_layers,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        ).to(self.device)


if __name__ == "__main__":
    model_file = "output/no-wandb-best_model.pth"
    ytid = "--aO5cdqSAg"
    config = AudioTokensConfig()
    tester = ManualTester(config, model_file)
    tester.run(ytid)
