import pandas as pd
import numpy as np

from get_card_columns import get_card_columns
from map_id_to_features import map_id_to_features

FEATURE_SUFFIXES = ['_wr', '_cmc', '_rarity']


def apply_feature_mapping(df, card_features, fallback_dict, global_defaults,
                          column_prefix, max_cards,
                          turn_wr_dict=None, use_dynamic_wr=False):
    """
    Replaces card ID columns with multi-attribute feature columns.

    For each card column like 'user_creatures_1', creates:
        - user_creatures_1_wr   (win rate)
        - user_creatures_1_cmc  (mana cost)
        - user_creatures_1_rarity (1-4 numeric)

    Then drops the original card ID column.

    Also adds zone-level aggregate features:
        - {prefix}_count  (number of non-empty card slots)
        - {prefix}_total_cmc (sum of CMC on board/in hand)

    Parameters:
        df: DataFrame with card ID columns
        card_features: dict from load_card_feature_mapping
        fallback_dict: dict from load_card_feature_mapping
        global_defaults: dict from load_card_feature_mapping
        column_prefix: e.g. 'user_creatures'
        max_cards: number of card slots (e.g. 20)
        turn_wr_dict: optional dict for dynamic turn-specific WR
        use_dynamic_wr: whether to use turn-specific WR

    Returns:
        pd.DataFrame with card ID columns replaced by feature columns
    """
    card_columns = get_card_columns(df, column_prefix, max_cards)

    for col in card_columns:
        # Determine turn for dynamic WR
        turn_for_lookup = None
        if use_dynamic_wr and turn_wr_dict is not None and 'turn' in df.columns:
            turn_for_lookup = 'per_row'  # sentinel to indicate per-row lookup

        wr_vals = []
        cmc_vals = []
        rarity_vals = []

        for idx, row in df.iterrows():
            card_id = row[col]
            turn = row.get('turn') if turn_for_lookup == 'per_row' else None

            feats = map_id_to_features(
                card_id, card_features, fallback_dict, global_defaults,
                turn_wr_dict=turn_wr_dict if use_dynamic_wr else None,
                turn=turn,
            )
            wr_vals.append(feats['wr'])
            cmc_vals.append(feats['cmc'])
            rarity_vals.append(feats['rarity'])

        df[f'{col}_wr'] = wr_vals
        df[f'{col}_cmc'] = cmc_vals
        df[f'{col}_rarity'] = rarity_vals

    # Drop original card ID columns
    df = df.drop(columns=card_columns, errors='ignore')

    # Add zone-level aggregates
    wr_cols = [f'{col}_wr' for col in card_columns if f'{col}_wr' in df.columns]
    cmc_cols = [f'{col}_cmc' for col in card_columns if f'{col}_cmc' in df.columns]

    if wr_cols:
        df[f'{column_prefix}_count'] = df[wr_cols].notna().sum(axis=1)
        df[f'{column_prefix}_avg_wr'] = df[wr_cols].mean(axis=1)
    if cmc_cols:
        df[f'{column_prefix}_total_cmc'] = df[cmc_cols].sum(axis=1)

    return df
