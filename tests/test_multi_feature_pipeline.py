"""
Tests for the multi-attribute card feature pipeline (v2).

Validates:
    - Card feature loading and fallback hierarchy
    - Per-card multi-attribute mapping
    - Zone-level feature expansion (apply_feature_mapping)
    - Dynamic turn-specific WR lookup
    - Unseen card fallback with rarity+CMC context
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'funcs'))


def test_load_card_feature_mapping():
    from load_card_feature_mapping import load_card_feature_mapping
    features, fallback, defaults = load_card_feature_mapping()

    assert len(features) > 10000, f"Expected 10k+ cards, got {len(features)}"
    assert len(fallback) > 20, f"Expected 20+ fallback buckets, got {len(fallback)}"
    assert 'wr' in defaults and 'cmc' in defaults and 'rarity' in defaults

    sample = list(features.values())[0]
    assert 'wr' in sample and 'cmc' in sample and 'rarity' in sample
    assert 0.3 < sample['wr'] < 0.8, f"WR out of range: {sample['wr']}"
    assert 0 <= sample['cmc'] <= 15
    assert sample['rarity'] in (1, 2, 3, 4)

    print("PASS: test_load_card_feature_mapping")


def test_map_id_to_features_known():
    from load_card_feature_mapping import load_card_feature_mapping
    from map_id_to_features import map_id_to_features

    features, fallback, defaults = load_card_feature_mapping()
    card_id = list(features.keys())[0]

    result = map_id_to_features(card_id, features, fallback, defaults)
    assert not np.isnan(result['wr'])
    assert not np.isnan(result['cmc'])
    assert not np.isnan(result['rarity'])
    print("PASS: test_map_id_to_features_known")


def test_map_id_to_features_unknown():
    from load_card_feature_mapping import load_card_feature_mapping
    from map_id_to_features import map_id_to_features

    features, fallback, defaults = load_card_feature_mapping()

    result = map_id_to_features(999999999, features, fallback, defaults)
    assert not np.isnan(result['wr']), "Fallback should provide a WR"
    print("PASS: test_map_id_to_features_unknown")


def test_map_id_to_features_nan():
    from load_card_feature_mapping import load_card_feature_mapping
    from map_id_to_features import map_id_to_features

    features, fallback, defaults = load_card_feature_mapping()

    result = map_id_to_features(np.nan, features, fallback, defaults)
    assert np.isnan(result['wr'])
    assert np.isnan(result['cmc'])
    print("PASS: test_map_id_to_features_nan")


def test_map_with_rarity_cmc_context():
    from load_card_feature_mapping import load_card_feature_mapping
    from map_id_to_features import map_id_to_features_with_context

    features, fallback, defaults = load_card_feature_mapping()

    result = map_id_to_features_with_context(
        999999999, features, fallback, defaults, rarity='rare', cmc=4
    )
    assert result['rarity'] == 3
    assert result['cmc'] == 4
    assert not np.isnan(result['wr'])
    print("PASS: test_map_with_rarity_cmc_context")


def test_apply_feature_mapping():
    from load_card_feature_mapping import load_card_feature_mapping
    from apply_feature_mapping import apply_feature_mapping

    features, fallback, defaults = load_card_feature_mapping()
    sample_ids = list(features.keys())[:2]

    df = pd.DataFrame({
        'turn': [3],
        'user_creatures_1': [str(sample_ids[0])],
        'user_creatures_2': [str(sample_ids[1])],
        'user_creatures_3': [np.nan],
    })

    result = apply_feature_mapping(
        df, features, fallback, defaults,
        column_prefix='user_creatures', max_cards=3,
    )

    # Original ID columns removed
    assert 'user_creatures_1' not in result.columns
    # New feature columns created
    assert 'user_creatures_1_wr' in result.columns
    assert 'user_creatures_1_cmc' in result.columns
    assert 'user_creatures_1_rarity' in result.columns
    # Aggregates created
    assert 'user_creatures_count' in result.columns
    assert 'user_creatures_avg_wr' in result.columns
    assert 'user_creatures_total_cmc' in result.columns
    # NaN slot should have NaN features
    assert np.isnan(result['user_creatures_3_wr'].iloc[0])
    # Count should be 2 (two non-NaN cards)
    assert result['user_creatures_count'].iloc[0] == 2

    print("PASS: test_apply_feature_mapping")


def test_dynamic_wr_turn_specific():
    from load_card_feature_mapping import load_card_feature_turn_mapping
    from apply_feature_mapping import apply_feature_mapping

    turn_wr_dict, features, fallback, defaults = load_card_feature_turn_mapping()

    # Card 91734 has turn-specific WR: turn 2 ~0.554, turn 5 ~0.533
    df = pd.DataFrame({
        'turn': [2, 5],
        'user_creatures_1': ['91734', '91734'],
        'user_creatures_2': [np.nan, np.nan],
    })

    result = apply_feature_mapping(
        df, features, fallback, defaults,
        column_prefix='user_creatures', max_cards=2,
        turn_wr_dict=turn_wr_dict, use_dynamic_wr=True,
    )

    wr_t2 = result['user_creatures_1_wr'].iloc[0]
    wr_t5 = result['user_creatures_1_wr'].iloc[1]
    assert abs(wr_t2 - 0.554) < 0.01
    assert abs(wr_t5 - 0.533) < 0.01
    assert wr_t2 > wr_t5

    print("PASS: test_dynamic_wr_turn_specific")


def test_mythic_vs_common_fallback():
    from load_card_feature_mapping import load_card_feature_mapping
    from map_id_to_features import map_id_to_features_with_context

    features, fallback, defaults = load_card_feature_mapping()

    mythic = map_id_to_features_with_context(
        999999999, features, fallback, defaults, rarity='mythic', cmc=4
    )
    common = map_id_to_features_with_context(
        999999999, features, fallback, defaults, rarity='common', cmc=4
    )
    assert mythic['wr'] > common['wr'], "Mythic should have higher avg WR than common"
    assert mythic['rarity'] == 4
    assert common['rarity'] == 1

    print("PASS: test_mythic_vs_common_fallback")


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    test_load_card_feature_mapping()
    test_map_id_to_features_known()
    test_map_id_to_features_unknown()
    test_map_id_to_features_nan()
    test_map_with_rarity_cmc_context()
    test_apply_feature_mapping()
    test_dynamic_wr_turn_specific()
    test_mythic_vs_common_fallback()

    print("\n=== ALL TESTS PASSED ===")
