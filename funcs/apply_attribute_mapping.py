"""
Replaces card ID/name columns with full attribute feature vectors.

For each card slot (e.g., user_creatures_1), creates ~35 feature columns.
Also computes zone-level aggregates that are dense and always populated:
    - {zone}_count: number of cards in zone
    - {zone}_total_power, {zone}_total_toughness: board presence
    - {zone}_total_cmc: mana investment
    - {zone}_avg_cmc: average card cost
    - {zone}_flyer_count: evasion
    - {zone}_removal_count: interaction (destroy + exile)
    - {zone}_planeswalker_count: planeswalker presence
"""
import numpy as np
import pandas as pd

from card_attributes import FEATURE_NAMES, empty_features
from get_card_columns import get_card_columns


def apply_attribute_mapping(df, resolver, column_prefix, max_cards):
    """
    Replace card ID columns with attribute feature columns.

    Parameters:
        df: DataFrame with card ID columns (e.g., user_creatures_1..20)
        resolver: CardFeatureResolver instance
        column_prefix: e.g., 'user_creatures'
        max_cards: number of card slots

    Returns:
        DataFrame with card ID columns replaced by per-slot features + zone aggregates
    """
    card_columns = get_card_columns(df, column_prefix, max_cards)

    # Extract features for each card in each slot
    all_slot_features = []  # list of list of feature dicts (per row, per slot)

    for _, row in df.iterrows():
        row_features = []
        for col in card_columns:
            card_val = row[col]

            if pd.isna(card_val) or card_val is None:
                row_features.append(empty_features())
                continue

            card_val_str = str(card_val).split('.')[0]  # handle "91734.0"

            # Try as numeric ID first, then as name
            try:
                int(card_val_str)
                feats = resolver.resolve(card_id=card_val_str)
            except ValueError:
                feats = resolver.resolve(card_name=card_val_str)

            row_features.append(feats)

        all_slot_features.append(row_features)

    # Build all new columns at once via dict (avoids DataFrame fragmentation)
    new_cols = {}
    for slot_idx, col in enumerate(card_columns):
        for feat_name in FEATURE_NAMES:
            new_col = f'{col}_{feat_name}'
            new_cols[new_col] = [
                row_feats[slot_idx][feat_name]
                for row_feats in all_slot_features
            ]

    new_cols_df = pd.DataFrame(new_cols, index=df.index)

    # Drop original card ID columns and concat new feature columns
    df = df.drop(columns=card_columns, errors='ignore')
    df = pd.concat([df, new_cols_df], axis=1)

    # Compute zone-level aggregates
    df = _add_zone_aggregates(df, column_prefix, card_columns)

    return df


def _add_zone_aggregates(df, prefix, original_card_columns):
    """Add dense zone-level aggregate features."""
    agg = {}

    def _sum_feat(feat_name):
        cols = [f'{col}_{feat_name}' for col in original_card_columns
                if f'{col}_{feat_name}' in df.columns]
        if not cols:
            return pd.Series(0, index=df.index)
        return df[cols].sum(axis=1)

    def _count_feat(feat_name, threshold=0.5):
        cols = [f'{col}_{feat_name}' for col in original_card_columns
                if f'{col}_{feat_name}' in df.columns]
        if not cols:
            return pd.Series(0, index=df.index)
        return (df[cols] > threshold).sum(axis=1)

    # Card count
    cmc_cols = [f'{col}_cmc' for col in original_card_columns
                if f'{col}_cmc' in df.columns]
    if cmc_cols:
        agg[f'{prefix}_count'] = df[cmc_cols].notna().sum(axis=1)
    else:
        agg[f'{prefix}_count'] = pd.Series(0, index=df.index)

    # Total stats
    agg[f'{prefix}_total_power'] = _sum_feat('power')
    agg[f'{prefix}_total_toughness'] = _sum_feat('toughness')
    agg[f'{prefix}_total_cmc'] = _sum_feat('cmc')
    agg[f'{prefix}_total_loyalty'] = _sum_feat('loyalty')

    # Average CMC
    count = agg[f'{prefix}_count'].replace(0, np.nan)
    agg[f'{prefix}_avg_cmc'] = agg[f'{prefix}_total_cmc'] / count

    # Keyword/ability counts
    agg[f'{prefix}_flyer_count'] = _count_feat('kw_flying')
    agg[f'{prefix}_deathtouch_count'] = _count_feat('kw_deathtouch')
    agg[f'{prefix}_lifelink_count'] = _count_feat('kw_lifelink')
    agg[f'{prefix}_haste_count'] = _count_feat('kw_haste')
    agg[f'{prefix}_hexproof_count'] = _count_feat('kw_hexproof')
    agg[f'{prefix}_indestructible_count'] = _count_feat('kw_indestructible')

    # Interaction counts
    agg[f'{prefix}_removal_count'] = _count_feat('has_destroy') + _count_feat('has_exile')
    agg[f'{prefix}_etb_count'] = _count_feat('has_etb')
    agg[f'{prefix}_draw_count'] = _count_feat('has_draw')

    # Type counts
    agg[f'{prefix}_planeswalker_count'] = _count_feat('is_planeswalker')
    agg[f'{prefix}_creature_count'] = _count_feat('is_creature')

    agg_df = pd.DataFrame(agg, index=df.index)
    df = pd.concat([df, agg_df], axis=1)

    return df
