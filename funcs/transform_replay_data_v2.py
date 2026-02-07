import pandas as pd
from transform_row_to_turns_v2 import transform_row_to_turns_v2
from load_card_feature_mapping import load_card_feature_mapping, load_card_feature_turn_mapping


def transform_replay_data_v2(data, max_cards=20, use_dynamic_wr=False):
    """
    Transforms replay data into turn-per-row format with multi-attribute
    card features (wr, cmc, rarity) and rarity+CMC fallback for unseen cards.

    Parameters:
        data: pd.DataFrame of raw replay data (game-per-row)
        max_cards: max card slots per zone
        use_dynamic_wr: use turn-specific WR for board creatures

    Returns:
        pd.DataFrame: one row per turn, with multi-attribute card features
    """
    # Load mappings once, pass to all rows
    if use_dynamic_wr:
        turn_wr_dict, card_features, fallback_dict, global_defaults = load_card_feature_turn_mapping()
    else:
        card_features, fallback_dict, global_defaults = load_card_feature_mapping()
        turn_wr_dict = None

    all_turns = []
    for idx, row in data.iterrows():
        turn_df = transform_row_to_turns_v2(
            row,
            card_features=card_features,
            fallback_dict=fallback_dict,
            global_defaults=global_defaults,
            turn_wr_dict=turn_wr_dict,
            max_cards=max_cards,
            use_dynamic_wr=use_dynamic_wr,
        )
        all_turns.append(turn_df)

    return pd.concat(all_turns, ignore_index=True)
