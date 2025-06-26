import os
import glob
import subprocess
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Process all pickle files in a directory")
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Directory containing .pkl files",
    )
    args = parser.parse_args()

    pkl_files = glob.glob(os.path.join(args.directory, "*.pkl"))

    for pkl_file in pkl_files:
        try:
            print(f"Processing file: {pkl_file}")
            subprocess.run(["python3", "saveParticleSize.py", "-i", pkl_file], check=False)
            print(f"Finished processing: {pkl_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {pkl_file}: {e}")


if __name__ == "__main__":
    main()
