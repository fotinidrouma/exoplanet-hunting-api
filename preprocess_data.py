"""
Exoplanet Hunting Dataset — Data Preprocessing
======================================================

Produces processed_exoTrain.csv and processed_exoTest.csv in DATA_DIR,
each with engineered features + the LABEL column.
"""

import os
import pandas as pd
from app.preprocessing import extract_features
import argparse

def preprocess_data(
        data_dir,
        train_fn,
        test_fn,
        output_prefix
):
    TRAIN_PATH = os.path.join(data_dir, train_fn)
    TEST_PATH = os.path.join(data_dir, test_fn)
    OUTPUT_TRAIN_PATH = os.path.join(data_dir, f"{output_prefix}_{train_fn}" if output_prefix else f"processed_{train_fn}")
    OUTPUT_TEST_PATH = os.path.join(data_dir, f"{output_prefix}_{test_fn}" if output_prefix else f"processed_{test_fn}")

    # 1. Load Data
    print("=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    flux_cols = [c for c in train.columns if c.startswith("FLUX")]
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # 2. Extract Features
    print("\n" + "=" * 60)
    print("2. EXTRACTING FEATURES")
    print("=" * 60)

    train_flux_array = train[flux_cols].values.astype(float)
    test_flux_array = test[flux_cols].values.astype(float)

    train_features = extract_features(train_flux_array)
    test_features = extract_features(test_flux_array)

    # Reattach the label
    train_features["LABEL"] = train["LABEL"].values
    test_features["LABEL"] = test["LABEL"].values

    print(f"Engineered feature columns: {[c for c in train_features.columns if c != 'LABEL']}")
    print(f"Processed train shape: {train_features.shape}")
    print(f"Processed test shape:  {test_features.shape}")

    # 3. Sanity Check — do the features actually separate the classes?
    print("\n" + "=" * 60)
    print("3. FEATURE SUMMARY BY CLASS (sanity check)")
    print("=" * 60)
    print(train_features.groupby("LABEL").mean().T)

    # 4. Save processed data
    print("\n" + "=" * 60)
    print("4. SAVING PROCESSED DATA")
    print("=" * 60)

    train_features.to_csv(OUTPUT_TRAIN_PATH, index=False)
    test_features.to_csv(OUTPUT_TEST_PATH, index=False)
    print(f"Saved -> {OUTPUT_TRAIN_PATH}")
    print(f"Saved -> {OUTPUT_TEST_PATH}")

if __name__ == '__main__':
    '''
    python preprocess_data.py \
        --data_dir ../data \
        --train_fn exoTrain.csv \
        --test_fn exoTest.csv
    '''

    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', type=str, help='input directory where files are', required=False, default='.')
    parser.add_argument('--train_fn', type=str, help='input train file', required=True)
    parser.add_argument('--test_fn', type=str, help='input test file', required=True)
    parser.add_argument('--output_prefix', type=str, help='output file prefix', required=False)

    args = parser.parse_args()

    preprocess_data(
        data_dir=args.data_dir,
        train_fn=args.train_fn,
        test_fn=args.test_fn,
        output_prefix=args.output_prefix
    )