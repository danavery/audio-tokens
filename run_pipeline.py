from audio_tokens_config import AudioTokensConfig
from processors.cluster_creator import ClusterCreator
from processors.model_trainer import ModelTrainer
from processors.spec_tokenizer import SpecTokenizer
from processors.spectrogram_generator import SpectrogramGenerator


def main():
    config = AudioTokensConfig()

    SpectrogramGenerator(config).run()
    ClusterCreator(config).run()
    SpecTokenizer(config).run()
    ModelTrainer(config).run()


if __name__ == "__main__":
    main()
