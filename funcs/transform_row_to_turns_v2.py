import pandas as pd
from explode_cards import explode_cards
from generate_card_columns import generate_card_columns
from count_player_lands import count_player_lands
from apply_feature_mapping import apply_feature_mapping
from load_card_feature_mapping import load_card_feature_mapping, load_card_feature_turn_mapping


def transform_row_to_turns_v2(row, card_features=None, fallback_dict=None,
                               global_defaults=None, turn_wr_dict=None,
                               max_cards=20, use_dynamic_wr=False):
    """
    Transforms a single game row into multiple turn rows with multi-attribute
    card features (wr, cmc, rarity) instead of a single WR scalar.

    Fallback for unseen cards: uses rarity+CMC bucket averages, then
    global defaults.

    Parameters:
        row: pd.Series from the raw replay dataset
        card_features: dict from load_card_feature_mapping (loaded once, passed in)
        fallback_dict: dict from load_card_feature_mapping
        global_defaults: dict from load_card_feature_mapping
        turn_wr_dict: optional dict for dynamic turn-specific WR
        max_cards: max card slots per zone
        use_dynamic_wr: use turn-specific WR for board creatures/non-creatures

    Returns:
        pd.DataFrame: one row per turn, card IDs replaced with feature columns
    """
    # Lazy-load mappings if not provided (for standalone usage)
    if card_features is None:
        if use_dynamic_wr:
            turn_wr_dict, card_features, fallback_dict, global_defaults = load_card_feature_turn_mapping()
        else:
            card_features, fallback_dict, global_defaults = load_card_feature_mapping()

    turns = []

    for turn in range(1, row['num_turns'] + 1):
        turn_data = {
            'game_id': row['draft_id'],
            'turn': turn,
            'on_play': row.get('on_play', None),
            'won': row.get('won', None),
        }
        try:
            turn_data['unique_id'] = row['unique_id']
        except Exception:
            pass

        for player in ['user', 'oppo']:
            # Cards in hand
            if player == "user":
                cards_in_hand_str = row.get(
                    f'{player}_turn_{turn}_eot_{player}_cards_in_hand', None)
                exploded = explode_cards(cards_in_hand_str, max_cards=max_cards)
                card_cols = generate_card_columns(exploded, f'{player}_hand')
                turn_data.update(card_cols)
            else:
                turn_data[f'{player}_cards_in_hand'] = row.get(
                    f'{player}_turn_{turn}_eot_{player}_cards_in_hand', None)

            # Lands
            land_values = row.get(
                f'{player}_turn_{turn}_eot_{player}_lands_in_play', None)
            turn_data[f'{player}_lands_in_play'] = count_player_lands(land_values)

            # Creatures
            creatures_str = row.get(
                f'{player}_turn_{turn}_eot_{player}_creatures_in_play', None)
            exploded_creatures = explode_cards(creatures_str, max_cards=max_cards)
            creature_cols = generate_card_columns(exploded_creatures, f'{player}_creatures')
            turn_data.update(creature_cols)

            # Non-creatures
            non_creatures_str = row.get(
                f'{player}_turn_{turn}_eot_{player}_non_creatures_in_play', None)
            exploded_nc = explode_cards(non_creatures_str, max_cards=max_cards)
            nc_cols = generate_card_columns(exploded_nc, f'{player}_non_creatures')
            turn_data.update(nc_cols)

            # Life
            turn_data[f'{player}_life'] = row.get(
                f'{player}_turn_{turn}_eot_{player}_life', None)

        turns.append(turn_data)

    turn_df = pd.DataFrame(turns)

    # Apply multi-feature mapping to all card zones
    zone_prefixes = [
        'user_hand', 'user_creatures', 'user_non_creatures',
        'oppo_creatures', 'oppo_non_creatures',
    ]
    for prefix in zone_prefixes:
        # Use dynamic WR only for board creatures/non-creatures
        use_dyn = use_dynamic_wr and 'creature' in prefix
        turn_df = apply_feature_mapping(
            turn_df,
            card_features=card_features,
            fallback_dict=fallback_dict,
            global_defaults=global_defaults,
            column_prefix=prefix,
            max_cards=max_cards,
            turn_wr_dict=turn_wr_dict if use_dyn else None,
            use_dynamic_wr=use_dyn,
        )

    return turn_df
