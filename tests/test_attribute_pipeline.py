"""
Tests for the v3 attribute-based card feature pipeline.

Validates:
    - Scryfall card DB loading
    - Feature extraction for all card types
    - CardFeatureResolver fallback hierarchy
    - apply_attribute_mapping zone expansion + aggregates
    - Unknown card handling (by minimal attributes)
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'funcs'))


def test_card_attributes_creature():
    from card_attributes import card_to_features, FEATURE_NAMES

    feats = card_to_features({
        'cmc': 4, 'type_line': 'Legendary Creature — Phyrexian Praetor',
        'oracle_text': 'Deathtouch\nWhenever you draw a card, you gain 2 life.',
        'power': 4, 'toughness': 5, 'loyalty': None,
        'colors': ['B'], 'color_identity': ['B'],
        'keywords': ['Deathtouch'], 'rarity': 'mythic',
    })

    assert feats['cmc'] == 4.0
    assert feats['power'] == 4.0
    assert feats['toughness'] == 5.0
    assert feats['is_creature'] == 1
    assert feats['is_black'] == 1
    assert feats['is_legendary'] == 1
    assert feats['kw_deathtouch'] == 1
    assert feats['rarity_num'] == 4
    assert feats['has_draw'] == 1
    assert len(feats) == len(FEATURE_NAMES)
    print("PASS: test_card_attributes_creature")


def test_card_attributes_planeswalker():
    from card_attributes import card_to_features

    feats = card_to_features({
        'cmc': 4, 'type_line': 'Legendary Planeswalker — Jace',
        'oracle_text': '+2: Look at the top card. -1: Return target creature. -12: Exile all cards from target library.',
        'power': None, 'toughness': None, 'loyalty': 3,
        'colors': ['U'], 'keywords': [], 'rarity': 'mythic',
    })

    assert feats['is_planeswalker'] == 1
    assert feats['is_creature'] == 0
    assert feats['loyalty'] == 3.0
    assert feats['is_blue'] == 1
    assert feats['has_exile'] == 1
    assert feats['has_activated'] == 1
    print("PASS: test_card_attributes_planeswalker")


def test_card_attributes_instant():
    from card_attributes import card_to_features

    feats = card_to_features({
        'cmc': 1, 'type_line': 'Instant',
        'oracle_text': 'Lightning Bolt deals 3 damage to any target.',
        'power': None, 'toughness': None, 'loyalty': None,
        'colors': ['R'], 'keywords': [], 'rarity': 'uncommon',
    })

    assert feats['is_instant'] == 1
    assert feats['is_creature'] == 0
    assert feats['is_red'] == 1
    assert feats['has_damage'] == 1
    assert feats['power'] == 0.0
    print("PASS: test_card_attributes_instant")


def test_card_attributes_artifact():
    from card_attributes import card_to_features

    feats = card_to_features({
        'cmc': 4, 'type_line': 'Legendary Artifact',
        'oracle_text': 'Indestructible\nWhen The One Ring enters the battlefield, gain protection. {1}, {T}: Draw cards.',
        'power': None, 'toughness': None, 'loyalty': None,
        'colors': [], 'keywords': ['Indestructible'], 'rarity': 'mythic',
    })

    assert feats['is_artifact'] == 1
    assert feats['is_colorless'] == 1
    assert feats['kw_indestructible'] == 1
    assert feats['has_etb'] == 1
    assert feats['has_draw'] == 1
    assert feats['has_activated'] == 1
    print("PASS: test_card_attributes_artifact")


def test_resolver_by_name():
    from card_feature_resolver import CardFeatureResolver

    resolver = CardFeatureResolver()
    if not resolver.scryfall_db:
        print("SKIP: test_resolver_by_name (no Scryfall data)")
        return

    feats = resolver.resolve(card_name='Lightning Bolt')
    assert feats['is_instant'] == 1
    assert feats['has_damage'] == 1
    assert feats['cmc'] == 1.0
    print("PASS: test_resolver_by_name")


def test_resolver_by_id():
    from card_feature_resolver import CardFeatureResolver

    resolver = CardFeatureResolver()
    if not resolver.id_to_name:
        print("SKIP: test_resolver_by_id (no cards.csv)")
        return

    # ID 91734 = Three Tree Rootweaver
    feats = resolver.resolve(card_id=91734)
    assert feats['is_creature'] == 1
    assert feats['cmc'] == 2.0
    print("PASS: test_resolver_by_id")


def test_resolver_unknown_with_hints():
    from card_feature_resolver import CardFeatureResolver

    resolver = CardFeatureResolver()

    feats = resolver.resolve(cmc=5, rarity='rare', card_type='creature',
                             colors=['R', 'G'], power=5, toughness=4)
    assert feats['cmc'] == 5.0
    assert feats['power'] == 5.0
    assert feats['toughness'] == 4.0
    assert feats['is_creature'] == 1
    assert feats['is_red'] == 1
    assert feats['is_green'] == 1
    assert feats['rarity_num'] == 3
    print("PASS: test_resolver_unknown_with_hints")


def test_resolver_fully_unknown():
    from card_feature_resolver import CardFeatureResolver

    resolver = CardFeatureResolver()

    feats = resolver.resolve()  # no args
    assert not np.isnan(feats['cmc']), "Global fallback should provide values"
    assert feats['cmc'] > 0
    print("PASS: test_resolver_fully_unknown")


def test_apply_attribute_mapping():
    from card_feature_resolver import CardFeatureResolver
    from apply_attribute_mapping import apply_attribute_mapping

    resolver = CardFeatureResolver()

    df = pd.DataFrame({
        'turn': [5],
        'user_creatures_1': ['91734'],  # Three Tree Rootweaver
        'user_creatures_2': [np.nan],
        'user_creatures_3': [np.nan],
    })

    result = apply_attribute_mapping(df, resolver, 'user_creatures', max_cards=3)

    # Original columns gone
    assert 'user_creatures_1' not in result.columns

    # Per-slot features exist
    assert 'user_creatures_1_cmc' in result.columns
    assert 'user_creatures_1_power' in result.columns
    assert 'user_creatures_1_is_creature' in result.columns

    # Zone aggregates exist
    assert 'user_creatures_count' in result.columns
    assert 'user_creatures_total_power' in result.columns
    assert 'user_creatures_total_toughness' in result.columns
    assert 'user_creatures_flyer_count' in result.columns

    # Count should be 1 (one non-NaN card)
    assert result['user_creatures_count'].iloc[0] == 1

    # NaN slot should have NaN features
    assert np.isnan(result['user_creatures_2_cmc'].iloc[0])

    print("PASS: test_apply_attribute_mapping")


def test_zone_aggregates_multicard():
    from card_feature_resolver import CardFeatureResolver
    from apply_attribute_mapping import apply_attribute_mapping

    resolver = CardFeatureResolver()
    if not resolver.scryfall_db:
        print("SKIP: test_zone_aggregates_multicard (no Scryfall data)")
        return

    # Two creatures: one with flying, one with deathtouch
    df = pd.DataFrame({
        'turn': [5],
        'user_creatures_1': ['atraxa, grand unifier'],   # 7/7 flying deathtouch lifelink vigilance
        'user_creatures_2': ['sheoldred, the apocalypse'],  # 4/5 deathtouch
        'user_creatures_3': [np.nan],
    })

    result = apply_attribute_mapping(df, resolver, 'user_creatures', max_cards=3)

    assert result['user_creatures_count'].iloc[0] == 2
    assert result['user_creatures_total_power'].iloc[0] == 11.0  # 7 + 4
    assert result['user_creatures_total_toughness'].iloc[0] == 12.0  # 7 + 5
    assert result['user_creatures_flyer_count'].iloc[0] >= 1  # Atraxa flies
    assert result['user_creatures_deathtouch_count'].iloc[0] == 2  # both

    print("PASS: test_zone_aggregates_multicard")


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    test_card_attributes_creature()
    test_card_attributes_planeswalker()
    test_card_attributes_instant()
    test_card_attributes_artifact()
    test_resolver_by_name()
    test_resolver_by_id()
    test_resolver_unknown_with_hints()
    test_resolver_fully_unknown()
    test_apply_attribute_mapping()
    test_zone_aggregates_multicard()

    print("\n=== ALL V3 ATTRIBUTE PIPELINE TESTS PASSED ===")
