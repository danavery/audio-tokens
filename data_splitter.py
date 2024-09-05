import logging
from pathlib import Path
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# YouTube ID character set (64 characters)
YOUTUBE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def split_data(audio_dir: Path, output_file: Path, val_size: int = 12):
    """
    Split the audio files into training and validation sets based on YouTube IDs.

    :param audio_dir: Directory containing the audio files
    :param output_file: JSON file to save the split information
    :param val_size: Integer from 0 to 63, representing the number of characters for validation
    """
    if not 0 <= val_size < 64:
        raise ValueError("val_size must be an integer between 0 and 63")

    all_files = [f for f in audio_dir.glob('**/*.flac')]  # Adjust file extension if needed

    train_files = []
    val_files = []

    val_chars = YOUTUBE_CHARS[:val_size]

    for file in all_files:
        # Assuming the YouTube ID is the filename without extension
        youtube_id = file.stem
        if youtube_id[-1] in val_chars:
            val_files.append(str(file.relative_to(audio_dir)))
        else:
            train_files.append(str(file.relative_to(audio_dir)))

    split_info = {
        'train': train_files,
        'validation': val_files
    }

    with open(output_file, 'w') as f:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        json.dump(split_info, f)

    logger.info(f"Total files: {len(all_files)}")
    logger.info(f"Training files: {len(train_files)}")
    logger.info(f"Validation files: {len(val_files)}")
    logger.info(f"Validation ratio: {val_size}/64 = {val_size/64:.2%}")
    logger.info(f"Split information saved to {output_file}")


if __name__ == "__main__":
    audio_dir = Path("/media/davery/audioset/bal_train")
    output_file = Path("output/bal_train_data_split.json")
    split_data(audio_dir, output_file, val_size=6)
