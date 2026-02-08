"""
Download 17Lands replay data for multiple sets.

Usage:
    python main/download_data.py                    # download all default sets
    python main/download_data.py --sets DSK FDN BLB # download specific sets
    python main/download_data.py --list             # show available sets
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Sets with known 17Lands replay data (Premier Draft)
# Recent sets tend to have the most data
AVAILABLE_SETS = {
    # 2024-2025
    'FDN': 'Foundations',
    'DSK': 'Duskmourn: House of Horror',
    'BLB': 'Bloomburrow',
    'OTJ': 'Outlaws of Thunder Junction',
    'MKM': 'Murders at Karlov Manor',
    # 2023
    'WOE': 'Wilds of Eldraine',
    'LCI': 'Lost Caverns of Ixalan',
    'MOM': 'March of the Machine',
    'ONE': 'Phyrexia: All Will Be One',
    # 2022
    'BRO': "The Brothers' War",
    'DMU': 'Dominaria United',
    'SNC': 'Streets of New Capenna',
    'NEO': 'Kamigawa: Neon Dynasty',
    # Older
    'VOW': 'Innistrad: Crimson Vow',
    'MID': 'Innistrad: Midnight Hunt',
    'AFR': 'Adventures in the Forgotten Realms',
    'STX': 'Strixhaven',
    'KHM': 'Kaldheim',
}

DEFAULT_SETS = ['DSK', 'FDN', 'BLB', 'OTJ', 'MKM', 'WOE', 'LCI']

BASE_URL = "https://17lands-public.s3.amazonaws.com/analysis_data/replay_data"


def download_set(set_code, output_dir, event_type='PremierDraft', max_retries=4):
    """Download a single set's replay data with retry logic."""
    filename = f"replay_data_public.{set_code}.{event_type}.csv.gz"
    url = f"{BASE_URL}/{filename}"
    output_path = output_dir / filename

    if output_path.exists():
        size_mb = output_path.stat().st_size / 1e6
        print(f"  {set_code}: already exists ({size_mb:.0f} MB), skipping")
        return True

    print(f"  {set_code}: downloading from {url} ...")

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['curl', '-sL', '-o', str(output_path), url],
                timeout=600,
                capture_output=True
            )
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                size_mb = output_path.stat().st_size / 1e6
                print(f"  {set_code}: done ({size_mb:.0f} MB)")
                return True
            else:
                print(f"  {set_code}: attempt {attempt+1} failed, retrying...")
                if output_path.exists():
                    output_path.unlink()
        except subprocess.TimeoutExpired:
            print(f"  {set_code}: timeout on attempt {attempt+1}")
            if output_path.exists():
                output_path.unlink()

        if attempt < max_retries - 1:
            import time
            wait = 2 ** (attempt + 1)
            print(f"  Waiting {wait}s before retry...")
            time.sleep(wait)

    print(f"  {set_code}: FAILED after {max_retries} attempts")
    return False


def main():
    parser = argparse.ArgumentParser(description='Download 17Lands replay data')
    parser.add_argument('--sets', nargs='+', default=None,
                        help=f'Set codes to download (default: {" ".join(DEFAULT_SETS)})')
    parser.add_argument('--list', action='store_true',
                        help='List available sets and exit')
    parser.add_argument('--all', action='store_true',
                        help='Download all available sets')
    parser.add_argument('--output-dir', type=str, default='data/raw_csv',
                        help='Output directory')
    args = parser.parse_args()

    if args.list:
        print("Available sets:")
        for code, name in AVAILABLE_SETS.items():
            marker = " (default)" if code in DEFAULT_SETS else ""
            print(f"  {code:>4}: {name}{marker}")
        return

    sets = list(AVAILABLE_SETS.keys()) if args.all else (args.sets or DEFAULT_SETS)

    # Validate set codes
    for s in sets:
        if s not in AVAILABLE_SETS:
            print(f"Warning: {s} not in known sets, attempting anyway")

    output_dir = Path(__file__).resolve().parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(sets)} sets to {output_dir}")
    print(f"Sets: {', '.join(sets)}")
    print()

    success = 0
    for set_code in sets:
        if download_set(set_code, output_dir):
            success += 1

    print(f"\nDone: {success}/{len(sets)} sets downloaded")

    # Show total size
    total = sum(f.stat().st_size for f in output_dir.glob('replay_data_public.*.csv.gz'))
    print(f"Total size: {total / 1e9:.1f} GB")


if __name__ == '__main__':
    main()
