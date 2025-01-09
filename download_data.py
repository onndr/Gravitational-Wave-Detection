"""Download dataset from Kaggle"""

import argparse
import os


KAGGLE_DATASET = "tentotheminus9/gravity-spy-gravitational-waves"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle dataset")
    parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--kaggle_username", required=True, help="Kaggle username")
    parser.add_argument("--kaggle_key", required=True, help="Kaggle key")

    args = parser.parse_args()

    os.environ['KAGGLE_USERNAME'] = args.kaggle_username
    os.environ['KAGGLE_KEY'] = args.kaggle_key

    import kaggle as kg

    kg.api.authenticate()
    kg.api.dataset_download_files(
        KAGGLE_DATASET,
        path=args.dataset_path,
        unzip=True
    )
