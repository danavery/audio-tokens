import matplotlib.pyplot as plt
from spectrogram_processor import SpectrogramProcessor, SpectrogramProcessorConfig


def verify_spectrogram_generation(audio_file_path):
    # Load configuration
    config = SpectrogramProcessorConfig(source_ytids=[])
    processor = SpectrogramProcessor(config)

    # Load and preprocess audio
    waveform = processor.preprocess_waveform(audio_file_path)

    # Generate spectrogram
    spec = processor.spec_generator.generate_mel_spectrogram(waveform)

    # Print spectrogram shape and statistics
    print(f"Spectrogram shape: {spec.shape}")
    print(f"Spectrogram min value: {spec.min():.4f}")
    print(f"Spectrogram max value: {spec.max():.4f}")
    print(f"Spectrogram mean value: {spec.mean():.4f}")

    # Visualize the spectrogram
    fig = processor.spec_generator.plot_spectrogram(
        config.common_sr, spec.T, processor.spec_generator.hop_length
    )
    plt.show()

    # If normalization is enabled, verify normalized spectrogram
    if config.normalize:
        norm_spec = processor.spec_generator.normalize_spectrogram(spec)
        print(f"\nNormalized spectrogram min value: {norm_spec.min():.4f}")
        print(f"Normalized spectrogram max value: {norm_spec.max():.4f}")
        print(f"Normalized spectrogram mean value: {norm_spec.mean():.4f}")

        fig = processor.spec_generator.plot_spectrogram(
            config.common_sr, norm_spec.T, processor.spec_generator.hop_length
        )
        plt.title("Normalized Spectrogram")
        plt.show()

    return spec


# Usage
audio_file_path = "/media/davery/audioset/bal_train/74/74MqoHo-kSs.flac"
spec = verify_spectrogram_generation(audio_file_path)
