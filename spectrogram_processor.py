import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Resample
from tqdm import tqdm

from audioset_metadata_processor import AudiosetMetadataProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class SpectrogramProcessorConfig:
    source_ytids: List[str]
    source_parent: str = "/media/davery/audioset"
    source_set: str = "bal_train"
    dest_path: str = "processed/"
    common_sr: int = 22050
    normalize: bool = True
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    n_segments: int = 0
    split_file: str = "output/bal_train_data_split.json"


class SpectrogramProcessor:
    def __init__(self, config):
        self.config = config
        self.spec_transformer = MelSpectrogram(
            sample_rate=self.config.common_sr,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        self.amplitude_to_db_transformer = AmplitudeToDB()
        self.logger = logging.getLogger()
        with open(config.split_file, 'r') as f:
            self.data_split = json.load(f)

    def run(self):
        for split in ['train', 'validation']:
            self.logger.info(f"Creating {split} spectrograms")
            specs = self.populate_specs(self.data_split[split])
            output_file = Path(self.config.dest_path) / f"{split}_specs.pkl"
            Path(self.config.dest_path).mkdir(exist_ok=True)
            with open(output_file, "wb") as f:
                pickle.dump(specs, f)
            self.logger.info(f"{split.capitalize()} spectrograms saved to: {output_file}")

    def populate_specs(self, source_files):
        specs = []
        for i, audio_file_path in enumerate(tqdm(source_files)):
            try:
                audio_file_path = f"{self.config.source_parent}/{self.config.source_set}/{audio_file_path}"
                waveform = self.preprocess_waveform(audio_file_path)
            except RuntimeError as e:
                if str(e) == "Failed to decode audio.":
                    self.logger.info(f"{e}: {audio_file_path}")
                    continue
                else:
                    raise

            spec = self.generate_mel_spectrogram(waveform)

            if self.config.normalize:
                spec = self.normalize_spectrogram(spec)
            if self.check_for_nan_inf(spec, f"spectrogram {i}"):
                self.logger.info(f"Bad file: {audio_file_path}")
                continue
            specs.append({"filename": os.path.basename(audio_file_path), "spec": spec})
        return specs

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

    def generate_mel_spectrogram(self, audio):
        mel_spec = self.spec_transformer(audio).squeeze(0)
        mel_spec_db = self.amplitude_to_db_transformer(mel_spec)
        return mel_spec_db

    @staticmethod
    def normalize_spectrogram(spec):
        spectrogram = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec))
        return spectrogram

    def check_for_nan_inf(self, data, name="data"):
        if torch.isnan(data).any():
            self.logger.info(f"Warning: NaN values found in {name}")
            self.logger.info(f"Indices of NaN values: {torch.nonzero(torch.isnan(data), as_tuple=True)}")
            return True
        if torch.isinf(data).any():
            self.logger.info(f"Warning: Inf values found in {name}")
            self.logger.info(f"Indices of NaN values: {torch.nonzero(torch.isinf(data), as_tuple=True)}")
            return True
        return False


if __name__ == "__main__":
    source_ytids = AudiosetMetadataProcessor().ytid_labels.keys()
    logging.getLogger().info(len(source_ytids))
    spec_processor_config = SpectrogramProcessorConfig(
        source_set="bal_train",
        source_ytids=source_ytids,
        n_segments=0,
    )
    SpectrogramProcessor(spec_processor_config).run()
