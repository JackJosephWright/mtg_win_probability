"""
Downloads Scryfall Oracle Cards bulk data and processes it into a local
card attribute database.

Oracle Cards = one row per unique card name (~30k cards, ~70MB JSON).
This covers every card ever printed in Magic.

Usage:
    python scryfall_loader.py            # download + process
    python scryfall_loader.py --refresh  # re-download
"""
import os
import json
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'scryfall')
ORACLE_CARDS_PATH = os.path.join(DATA_DIR, 'oracle_cards.json')
PROCESSED_PATH = os.path.join(DATA_DIR, 'card_attributes.json')
BULK_DATA_INDEX = "https://api.scryfall.com/bulk-data"


def get_oracle_download_url():
    """Fetch the current Oracle Cards download URL from Scryfall."""
    req = urllib.request.Request(BULK_DATA_INDEX, headers={'User-Agent': 'MTGWinProbBot/1.0'})
    with urllib.request.urlopen(req) as resp:
        bulk_data = json.loads(resp.read().decode())
    for item in bulk_data['data']:
        if item['type'] == 'oracle_cards':
            return item['download_uri']
    raise RuntimeError("Could not find oracle_cards in Scryfall bulk data index")


def download_oracle_cards(force=False):
    """Download Oracle Cards JSON from Scryfall."""
    if os.path.exists(ORACLE_CARDS_PATH) and not force:
        print(f"Already downloaded: {ORACLE_CARDS_PATH}")
        return ORACLE_CARDS_PATH

    os.makedirs(DATA_DIR, exist_ok=True)
    url = get_oracle_download_url()
    print(f"Downloading Oracle Cards from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'MTGWinProbBot/1.0'})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    with open(ORACLE_CARDS_PATH, 'wb') as f:
        f.write(data)
    print(f"Saved {len(data) / 1e6:.1f}MB to {ORACLE_CARDS_PATH}")
    return ORACLE_CARDS_PATH


def process_oracle_cards(oracle_path=None):
    """
    Process raw Scryfall JSON into a compact card attribute dict.

    Returns:
        dict: {card_name_lower: {attributes...}} for every card
    """
    oracle_path = oracle_path or ORACLE_CARDS_PATH
    with open(oracle_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    print(f"Processing {len(cards)} cards...")
    card_db = {}

    for card in cards:
        # Skip tokens, emblems, art series, etc.
        layout = card.get('layout', '')
        if layout in ('token', 'art_series', 'double_faced_token', 'emblem'):
            continue

        name = card.get('name', '')
        if not name:
            continue

        # For double-faced cards, use the front face
        if 'card_faces' in card and card['card_faces']:
            face = card['card_faces'][0]
            oracle_text = face.get('oracle_text', '')
            power = face.get('power', None)
            toughness = face.get('toughness', None)
            loyalty = face.get('loyalty', None)
            type_line = face.get('type_line', card.get('type_line', ''))
        else:
            oracle_text = card.get('oracle_text', '')
            power = card.get('power', None)
            toughness = card.get('toughness', None)
            loyalty = card.get('loyalty', None)
            type_line = card.get('type_line', '')

        # Parse power/toughness (handle *, X, etc.)
        power_num = _parse_pt(power)
        toughness_num = _parse_pt(toughness)
        loyalty_num = _parse_pt(loyalty)

        attrs = {
            'name': name,
            'cmc': card.get('cmc', 0),
            'type_line': type_line,
            'oracle_text': oracle_text,
            'power': power_num,
            'toughness': toughness_num,
            'loyalty': loyalty_num,
            'colors': card.get('colors', []),
            'color_identity': card.get('color_identity', []),
            'keywords': card.get('keywords', []),
            'rarity': card.get('rarity', 'common'),
        }

        # Index by lowercase name for fuzzy matching
        card_db[name.lower()] = attrs

    print(f"Processed {len(card_db)} unique cards")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PROCESSED_PATH, 'w') as f:
        json.dump(card_db, f)
    print(f"Saved to {PROCESSED_PATH}")

    return card_db


def load_card_db():
    """Load the processed card attribute database."""
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Card database not found at {PROCESSED_PATH}. "
            "Run `python scryfall_loader.py` first."
        )
    with open(PROCESSED_PATH, 'r') as f:
        return json.load(f)


def _parse_pt(val):
    """Parse power/toughness/loyalty to float. Handles *, X, 1+*, etc."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        # * = variable (treat as 0), X = variable
        if val in ('*', 'X', '?'):
            return 0.0
        # Handle "1+*", "2+*" etc.
        if '+' in str(val):
            try:
                return float(str(val).split('+')[0])
            except ValueError:
                return 0.0
        return 0.0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Re-download data')
    args = parser.parse_args()

    download_oracle_cards(force=args.refresh)
    db = process_oracle_cards()
    print(f"\nSample entries:")
    for name in ['lightning bolt', 'sheoldred, the apocalypse', 'jace, the mind sculptor']:
        if name in db:
            c = db[name]
            print(f"  {c['name']}: cmc={c['cmc']}, p/t={c['power']}/{c['toughness']}, "
                  f"loyalty={c['loyalty']}, keywords={c['keywords']}, rarity={c['rarity']}")
