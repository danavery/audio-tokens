"""
AudioSet Expander Script

Extracts and organizes audio files from AudioSet tar archives into a structured
directory hierarchy. Processes tar files in the current directory, organizing
FLAC files into unbalanced train, evaluation, and balanced train sets.

"""


import os
import tarfile
import shutil
import tempfile

# Base destination directory where the files will be organized
BASE_DEST_DIR = "/media/davery/audioset"

# Create directories for the three sets
SET1_DIR = os.path.join(BASE_DEST_DIR, "unbal_train")
SET2_DIR = os.path.join(BASE_DEST_DIR, "eval")
SET3_DIR = os.path.join(BASE_DEST_DIR, "bal_train")

# Ensure the base directories exist
os.makedirs(SET1_DIR, exist_ok=True)
os.makedirs(SET2_DIR, exist_ok=True)
os.makedirs(SET3_DIR, exist_ok=True)


def determine_set(tarfile_name):
    """Determine the set based on the tarfile name."""
    if "unbal_train" in tarfile_name:
        return SET1_DIR
    elif "eval" in tarfile_name:
        return SET2_DIR
    else:
        return SET3_DIR


def extract_and_organize(tarfile_path):
    """Extract and organize files from a tar archive."""
    tarfile_name = os.path.basename(tarfile_path)
    dest_dir = determine_set(tarfile_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(tarfile_path, "r:*") as tar:
            tar.extractall(path=temp_dir)

        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".flac"):
                    subdir = file[:2]
                    dest_subdir = os.path.join(dest_dir, subdir)
                    os.makedirs(dest_subdir, exist_ok=True)
                    shutil.move(os.path.join(root, file), dest_subdir)


if __name__ == "__main__":
    tar_files = [
        f
        for f in os.listdir(".")
        if f.endswith(".tar") or f.endswith(".tar.gz") or f.endswith(".tar.bz2")
    ]

    for tfile in tar_files:
        print(f"Processing {tfile}...")
        extract_and_organize(tfile)
        print(f"Finished processing {tfile}.")
