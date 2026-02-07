import numpy as np


def map_id_to_features(card_id, card_features, fallback_dict, global_defaults,
                       turn_wr_dict=None, turn=None):
    """
    Maps a card ID to its feature vector: (wr, cmc, rarity).

    Fallback hierarchy:
        1. Known card ID -> exact features (with optional turn-specific WR)
        2. Unknown card but rarity+CMC bucket exists -> bucket average
        3. Completely unknown -> global defaults

    Parameters:
        card_id: raw card ID value (str or int or float)
        card_features: dict from load_card_feature_mapping
        fallback_dict: dict from load_card_feature_mapping
        global_defaults: dict from load_card_feature_mapping
        turn_wr_dict: optional dict of turn-specific WR {"cardid_turn": wr}
        turn: optional int turn number for dynamic WR lookup

    Returns:
        dict: {'wr': float, 'cmc': float, 'rarity': float}
    """
    if card_id is None or (isinstance(card_id, float) and np.isnan(card_id)):
        return {'wr': np.nan, 'cmc': np.nan, 'rarity': np.nan}

    # Normalize card_id to int
    try:
        card_id_int = int(float(str(card_id).split('.')[0]))
    except (ValueError, TypeError):
        return {'wr': np.nan, 'cmc': np.nan, 'rarity': np.nan}

    # Level 1: Known card
    if card_id_int in card_features:
        feats = card_features[card_id_int].copy()

        # Try turn-specific WR override
        if turn_wr_dict is not None and turn is not None:
            turn_key = f"{card_id_int}_{turn}"
            turn_wr = turn_wr_dict.get(turn_key)
            if turn_wr is not None:
                feats['wr'] = turn_wr

        return feats

    # Level 2: Fallback to rarity+CMC bucket
    # We don't know this card's rarity/CMC directly, so use global defaults
    # (In practice, if a card is in cards.csv but not in ratings, it would
    # still be in card_features with NaN wr. This branch handles cards
    # completely absent from training data.)
    for rarity_guess in [1, 2, 3]:  # try common first, then uncommon, rare
        bucket_key = (rarity_guess, global_defaults['cmc'])
        if bucket_key in fallback_dict:
            return fallback_dict[bucket_key].copy()

    # Level 3: Global average
    return global_defaults.copy()


def map_id_to_features_with_context(card_id, card_features, fallback_dict,
                                     global_defaults, rarity=None, cmc=None,
                                     turn_wr_dict=None, turn=None):
    """
    Like map_id_to_features but accepts optional rarity/cmc hints for
    unknown cards (e.g. from a bot that knows the card's printed attributes).

    This lets the bot provide rarity+CMC for cards not in training data,
    giving the model a better fallback than the global average.
    """
    if card_id is None or (isinstance(card_id, float) and np.isnan(card_id)):
        return {'wr': np.nan, 'cmc': np.nan, 'rarity': np.nan}

    try:
        card_id_int = int(float(str(card_id).split('.')[0]))
    except (ValueError, TypeError):
        return {'wr': np.nan, 'cmc': np.nan, 'rarity': np.nan}

    # Known card - use exact features
    if card_id_int in card_features:
        feats = card_features[card_id_int].copy()
        if turn_wr_dict is not None and turn is not None:
            turn_key = f"{card_id_int}_{turn}"
            turn_wr = turn_wr_dict.get(turn_key)
            if turn_wr is not None:
                feats['wr'] = turn_wr
        return feats

    # Unknown card - use provided rarity+CMC hints for better fallback
    from load_card_feature_mapping import RARITY_MAP
    rarity_num = RARITY_MAP.get(rarity, rarity) if rarity is not None else None

    if rarity_num is not None and cmc is not None:
        bucket_key = (int(rarity_num), int(cmc))
        if bucket_key in fallback_dict:
            return fallback_dict[bucket_key].copy()
        # Rarity known but exact CMC bucket missing - try nearby CMC
        for cmc_offset in [1, -1, 2, -2]:
            nearby_key = (int(rarity_num), int(cmc) + cmc_offset)
            if nearby_key in fallback_dict:
                result = fallback_dict[nearby_key].copy()
                result['cmc'] = int(cmc)  # use actual CMC, not the nearby one
                return result

    return global_defaults.copy()
