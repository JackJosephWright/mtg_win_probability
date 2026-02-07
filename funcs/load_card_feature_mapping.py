import pandas as pd
import numpy as np


RARITY_MAP = {'common': 1, 'uncommon': 2, 'rare': 3, 'mythic': 4, 'basic': 1,
              'C': 1, 'U': 2, 'R': 3, 'M': 4, 'B': 1}


def load_card_feature_mapping():
    """
    Loads card features into a dict: card_id -> {wr, cmc, rarity}.

    Also builds a rarity+CMC fallback table for unseen cards.

    Returns:
        tuple: (card_features_dict, fallback_dict, global_defaults)
            - card_features_dict: {int(card_id): {'wr': float, 'cmc': int, 'rarity': int}}
            - fallback_dict: {(rarity_num, cmc): {'wr': float, 'cmc': int, 'rarity': int}}
            - global_defaults: {'wr': float, 'cmc': float, 'rarity': float}
    """
    cards = pd.read_csv("data/cards_data/cards.csv")
    ratings = pd.read_csv("data/cards_data/combined_card_ratings.csv")

    merged = cards.merge(ratings, left_on='name', right_on='Name')

    merged['gp_wr'] = merged['GP WR'].str.rstrip('%').astype(float) / 100.0
    merged['rarity_num'] = merged['rarity'].map(RARITY_MAP).fillna(1).astype(int)
    merged['is_creature'] = merged['types'].str.contains('Creature', na=False).astype(int)

    # Deduplicate: keep the row with highest sample count per card id
    merged['gp_count'] = merged['# GP'].fillna(0)
    merged = merged.sort_values('gp_count', ascending=False).drop_duplicates(subset='id', keep='first')

    # Build per-card feature dict
    card_features = {}
    for _, row in merged.iterrows():
        card_id = int(row['id'])
        card_features[card_id] = {
            'wr': row['gp_wr'] if pd.notnull(row['gp_wr']) else np.nan,
            'cmc': int(row['mana_value']) if pd.notnull(row['mana_value']) else 0,
            'rarity': row['rarity_num'],
        }

    # Build rarity+CMC fallback from aggregate stats
    valid = merged[merged['gp_wr'].notnull()]
    fallback_dict = {}
    for (rarity_num, cmc), group in valid.groupby(['rarity_num', 'mana_value']):
        fallback_dict[(int(rarity_num), int(cmc))] = {
            'wr': group['gp_wr'].mean(),
            'cmc': int(cmc),
            'rarity': int(rarity_num),
        }

    # Global defaults for completely unknown cards
    global_defaults = {
        'wr': valid['gp_wr'].mean(),
        'cmc': valid['mana_value'].mean(),
        'rarity': 1.0,
    }

    return card_features, fallback_dict, global_defaults


def load_card_feature_turn_mapping():
    """
    Loads turn-specific win rates and merges with card attributes.

    Returns:
        tuple: (turn_wr_dict, card_features, fallback_dict, global_defaults)
            - turn_wr_dict: {"cardid_turn": wr_float}
            - card_features: same as load_card_feature_mapping
            - fallback_dict: same as load_card_feature_mapping
            - global_defaults: same as load_card_feature_mapping
    """
    wr_by_turn = pd.read_csv("./data/card_wr_by_turn_dict/card_wr_by_turn_total_df.csv")
    wr_by_turn['key'] = wr_by_turn['Card ID'].astype(int).astype(str) + "_" + wr_by_turn['Turn'].astype(str)
    turn_wr_dict = wr_by_turn[['key', 'Win Rate']].set_index('key')['Win Rate'].to_dict()

    card_features, fallback_dict, global_defaults = load_card_feature_mapping()

    return turn_wr_dict, card_features, fallback_dict, global_defaults


if __name__ == '__main__':
    features, fallback, defaults = load_card_feature_mapping()
    print(f"Loaded {len(features)} card features")
    print(f"Fallback buckets: {len(fallback)}")
    print(f"Global defaults: {defaults}")
    # Show a sample
    sample_id = list(features.keys())[0]
    print(f"Sample card {sample_id}: {features[sample_id]}")
