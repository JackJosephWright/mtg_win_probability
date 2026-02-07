"""
v3 Transform: Full attribute-based card features for any format.

Changes from v2:
    - Uses Scryfall-derived card attributes (~35 features per card) instead of WR scalar
    - Works for ANY card ever printed (not limited to 17Lands training set)
    - Handles all card types: creatures, planeswalkers, instants, sorceries, etc.
    - Zone-level aggregates (total power, flyer count, removal count, etc.)
    - Fallback for completely unknown cards via minimal attributes

For the bot: call CardFeatureResolver.resolve() with whatever you know about the card.
"""
import pandas as pd
from explode_cards import explode_cards
from generate_card_columns import generate_card_columns
from count_player_lands import count_player_lands
from apply_attribute_mapping import apply_attribute_mapping
from card_feature_resolver import CardFeatureResolver


def transform_row_to_turns_v3(row, resolver=None, max_cards=10):
    """
    Transform a single game row into turn rows with full attribute features.

    Parameters:
        row: pd.Series from raw replay data
        resolver: CardFeatureResolver instance (loaded once, reused)
        max_cards: max card slots per zone (10 is enough for most games)

    Returns:
        pd.DataFrame: one row per turn with attribute feature columns
    """
    if resolver is None:
        resolver = CardFeatureResolver()

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
                hand_str = row.get(
                    f'{player}_turn_{turn}_eot_{player}_cards_in_hand', None)
                exploded = explode_cards(hand_str, max_cards=max_cards)
                turn_data.update(generate_card_columns(exploded, f'{player}_hand'))
            else:
                turn_data[f'{player}_cards_in_hand'] = row.get(
                    f'{player}_turn_{turn}_eot_{player}_cards_in_hand', None)

            # Lands (count only)
            land_str = row.get(
                f'{player}_turn_{turn}_eot_{player}_lands_in_play', None)
            turn_data[f'{player}_lands_in_play'] = count_player_lands(land_str)

            # Creatures
            creatures_str = row.get(
                f'{player}_turn_{turn}_eot_{player}_creatures_in_play', None)
            exploded_c = explode_cards(creatures_str, max_cards=max_cards)
            turn_data.update(generate_card_columns(exploded_c, f'{player}_creatures'))

            # Non-creatures (includes planeswalkers, artifacts, enchantments)
            nc_str = row.get(
                f'{player}_turn_{turn}_eot_{player}_non_creatures_in_play', None)
            exploded_nc = explode_cards(nc_str, max_cards=max_cards)
            turn_data.update(generate_card_columns(exploded_nc, f'{player}_non_creatures'))

            # Life
            turn_data[f'{player}_life'] = row.get(
                f'{player}_turn_{turn}_eot_{player}_life', None)

        turns.append(turn_data)

    turn_df = pd.DataFrame(turns)

    # Apply attribute feature mapping to all card zones
    for prefix in ['user_hand', 'user_creatures', 'user_non_creatures',
                    'oppo_creatures', 'oppo_non_creatures']:
        turn_df = apply_attribute_mapping(turn_df, resolver, prefix, max_cards)

    return turn_df


def transform_replay_data_v3(data, max_cards=10):
    """
    Batch transform: converts raw replay data into turn-level attribute features.

    Parameters:
        data: pd.DataFrame of raw replay data (game-per-row)
        max_cards: max card slots per zone

    Returns:
        pd.DataFrame: one row per turn with attribute features
    """
    resolver = CardFeatureResolver()

    all_turns = []
    for idx, row in data.iterrows():
        turn_df = transform_row_to_turns_v3(row, resolver=resolver, max_cards=max_cards)
        all_turns.append(turn_df)

    return pd.concat(all_turns, ignore_index=True)
