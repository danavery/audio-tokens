import json
import logging
import os
import numpy as np
from pathlib import Path
import shutil

import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Resample
from tqdm import tqdm

from audio_tokens_config import AudioTokensConfig


class SpectrogramProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.spec_transformer = MelSpectrogram(
            sample_rate=self.config.common_sr,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        ).to(self.device)
        self.amplitude_to_db_transformer = AmplitudeToDB().to(self.device)

        with open(config.split_file, "r") as f:
            self.data_split = json.load(f)

    def run(self):
        for split in ["train", "validation"]:
            self.logger.info(f"Creating {split} spectrograms")
            output_dir = Path(self.config.dest_spec_path) / split
            shutil.rmtree(output_dir, ignore_errors=True)
            output_dir.mkdir(parents=True)

            ytids = self.data_split[split]
            for i in tqdm(
                range(0, len(ytids), self.config.spectrogram_batch_size),
                total=len(ytids) // self.config.spectrogram_batch_size,
                position=0,
            ):
                batch_ytids = ytids[i : i + self.config.spectrogram_batch_size]
                specs = self.populate_specs(batch_ytids)

                for spec in specs:
                    ytid = spec["filename"].replace(".flac", "")
                    output_file = output_dir / f"{ytid}.npy"
                    np.save(output_file, spec["spec"].cpu())
            self.logger.info(
                f"{split.capitalize()} spectrograms saved to: {output_dir}"
            )

    def populate_specs(self, source_files):
        specs = []
        for i, ytid in tqdm(
            enumerate(source_files), position=1, total=len(source_files)
        ):

            audio_file_path = self.find_audio_file(ytid)
            if not audio_file_path:
                continue
            waveform = self.preprocess_waveform(audio_file_path)
            if waveform is None:
                continue
            spec = self.generate_mel_spectrogram(waveform)

            if self.config.normalize:
                spec = self.normalize_spectrogram(spec)

            if self.check_for_nan_inf(spec, f"spectrogram {i}"):
                self.logger.debug(f"Bad file: {audio_file_path}")
                continue

            specs.append({"filename": os.path.basename(audio_file_path), "spec": spec})
        return specs

    def find_audio_file(self, ytid):
        for source_set in self.config.audio_source_sets:
            audio_file_path = Path(f"{self.config.audio_source_path}/{source_set}/{ytid[:2]}/{ytid}.flac")
            if audio_file_path.exists():
                return audio_file_path
        self.logger.debug(f"Audio file not found: {audio_file_path}")
        return None

    def preprocess_waveform(self, audio_file_path):
        try:
            waveform, sr = torchaudio.load(audio_file_path)
        except RuntimeError as e:
            if (str(e) == "Failed to decode audio."):
                self.logger.info(f"skipping {audio_file_path}: {e}")
                return None
        waveform = waveform.to(self.device)
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
        if sr != self.config.common_sr:
            resampler = Resample(sr, self.config.common_sr).to(self.device)
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


if __name__ == "__main__":
    config = AudioTokensConfig()
    SpectrogramProcessor(config).run()
