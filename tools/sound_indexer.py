import os
import tarfile


def process_tar_files(directories):
    with open("ytid_dir.csv", "w") as output:
        for directory in directories:
            print(directory)
            # List all tar files in the specified directory
            tar_files = [f for f in os.listdir(directory) if f.endswith(".tar")]

            # Process each tar file
            for tar_file in tar_files:
                # Construct the full path to the tar file
                tar_path = os.path.join(directory, tar_file)

                # Open the tar file
                with tarfile.open(tar_path, "r") as tar:
                    # List all members in the tar file
                    for member in tar.getmembers():
                        # Extract each member (optional)
                        ytid, filepath = (
                            os.path.splitext(os.path.basename(member.name))[0],
                            tar_path,
                        )
                        print(f"{ytid},{filepath}", file=output)
                # print(f"Processed {tar_file}")


process_tar_files(
    [
        "/home/davery/audioset-hf/audioset-dl/data",
        "/media/davery/T7/AudioSet/data",
        "/media/davery/airport/AudioSet/data",
    ]
)
