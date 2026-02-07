"""
Process raw 17Lands replay data into multi-attribute turn-level features.

v2: Uses (wr, cmc, rarity) per card slot instead of WR-only.
    Includes rarity+CMC fallback for unseen cards.

Usage:
    python gen_all_v2.py [--nrows N] [--chunk_size N] [--dynamic]
"""
import os
import sys
import hashlib
import argparse
import pandas as pd

# Add funcs to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
funcs_dir = os.path.join(project_root, 'funcs')
if funcs_dir not in sys.path:
    sys.path.insert(0, funcs_dir)

from transform_replay_data_v2 import transform_replay_data_v2

INPUT_DIR = os.path.join(project_root, 'data', 'raw_csv')
OUTPUT_PATH = os.path.join(project_root, 'data', 'processed_csv', 'multi_feature_by_turn.csv')


def generate_unique_id(row):
    unique_string = f"{row['draft_id']}-{row['opening_hand']}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def process_file(input_file, save_path, nrows=100000, chunk_size=5000,
                 use_dynamic_wr=False):
    print(f"Loading {input_file}...")
    data = pd.read_csv(input_file, nrows=nrows)
    data['unique_id'] = data.apply(generate_unique_id, axis=1)

    # Skip already-processed rows
    processed_ids = set()
    if os.path.exists(save_path):
        processed_ids = set(pd.read_csv(save_path, usecols=['unique_id'])['unique_id'])
        print(f"Skipping {len(processed_ids)} already-processed games")

    unprocessed = data[~data['unique_id'].isin(processed_ids)]
    total = len(unprocessed)
    print(f"Rows to process: {total}")

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = unprocessed.iloc[start:end]
        print(f"  Processing rows {start+1}-{end}...")

        output = transform_replay_data_v2(
            chunk, max_cards=20, use_dynamic_wr=use_dynamic_wr
        )

        if os.path.exists(save_path):
            output.to_csv(save_path, mode='a', index=False, header=False)
        else:
            output.to_csv(save_path, mode='w', index=False, header=True)

    print(f"Done. Output: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrows', type=int, default=100000)
    parser.add_argument('--chunk_size', type=int, default=5000)
    parser.add_argument('--dynamic', action='store_true',
                        help='Use turn-specific WR for board creatures')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    for fname in sorted(os.listdir(INPUT_DIR)):
        if fname.endswith('.csv.gz') or fname.endswith('.csv'):
            process_file(
                os.path.join(INPUT_DIR, fname),
                OUTPUT_PATH,
                nrows=args.nrows,
                chunk_size=args.chunk_size,
                use_dynamic_wr=args.dynamic,
            )

    print("\nAll files processed.")


if __name__ == '__main__':
    main()
