"""
Tests for the turn-specific WR meta-model estimator.

Validates:
    - Meta-model loads and produces predictions
    - Predictions vary by turn (temporal dynamics exist)
    - Hand penalty: expensive cards in hand early are penalized
    - Known cards get reasonable estimates
    - Unknown cards with minimal attrs get estimates
    - Integration with CardFeatureResolver.resolve_with_turn_wr
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'funcs'))


def test_estimator_loads():
    from turn_wr_estimator import TurnWREstimator
    estimator = TurnWREstimator()
    assert estimator.model is not None
    print("PASS: test_estimator_loads")


def test_estimate_varies_by_turn():
    from turn_wr_estimator import TurnWREstimator
    estimator = TurnWREstimator()

    # A 2-drop creature should have different WR on turn 2 vs turn 10
    attrs = {
        'cmc': 2, 'type_line': 'Creature', 'oracle_text': '',
        'power': 2, 'toughness': 2, 'loyalty': 0,
        'colors': ['G'], 'keywords': [], 'rarity': 'common',
    }
    wr_t2 = estimator.estimate(attrs, turn=2)
    wr_t10 = estimator.estimate(attrs, turn=10)

    assert 0.3 < wr_t2 < 0.8, f"WR out of range: {wr_t2}"
    assert 0.3 < wr_t10 < 0.8, f"WR out of range: {wr_t10}"
    assert wr_t2 != wr_t10, "WR should vary by turn"
    print(f"PASS: test_estimate_varies_by_turn (turn2={wr_t2:.3f}, turn10={wr_t10:.3f})")


def test_estimate_curve():
    from turn_wr_estimator import TurnWREstimator
    estimator = TurnWREstimator()

    attrs = {
        'cmc': 4, 'type_line': 'Creature', 'oracle_text': 'Flying',
        'power': 4, 'toughness': 4, 'loyalty': 0,
        'colors': ['W'], 'keywords': ['Flying'], 'rarity': 'rare',
    }
    curve = estimator.estimate_curve(attrs, max_turn=10)

    assert len(curve) == 10
    assert all(0.3 < v < 0.9 for v in curve.values()), f"WR values out of range: {curve}"
    print(f"PASS: test_estimate_curve (turns 1-10: {[f'{v:.3f}' for v in curve.values()]})")


def test_hand_penalty_expensive_card():
    from turn_wr_estimator import TurnWREstimator
    estimator = TurnWREstimator()

    attrs = {
        'cmc': 7, 'type_line': 'Creature', 'oracle_text': 'Flying trample',
        'power': 7, 'toughness': 7, 'loyalty': 0,
        'colors': ['G', 'W'], 'keywords': ['Flying', 'Trample'], 'rarity': 'mythic',
    }

    # Turn 1: 7-drop stuck in hand (6 turns away from castable)
    penalty_t1 = estimator.estimate_hand_penalty(attrs, turn=1)
    assert penalty_t1 < 0.5, f"Should be penalized: {penalty_t1}"

    # Turn 7: 7-drop is castable
    castable_t7 = estimator.estimate_hand_penalty(attrs, turn=7)
    assert castable_t7 > 0.5, f"Should be positive: {castable_t7}"

    assert castable_t7 > penalty_t1, "Castable should be better than stuck"
    print(f"PASS: test_hand_penalty_expensive_card (t1={penalty_t1:.3f}, t7={castable_t7:.3f})")


def test_hand_penalty_cheap_card():
    from turn_wr_estimator import TurnWREstimator
    estimator = TurnWREstimator()

    attrs = {
        'cmc': 1, 'type_line': 'Instant', 'oracle_text': 'Deals 3 damage',
        'power': 0, 'toughness': 0, 'loyalty': 0,
        'colors': ['R'], 'keywords': [], 'rarity': 'common',
    }

    # 1-drop should always be castable
    wr_t1 = estimator.estimate_hand_penalty(attrs, turn=1)
    wr_t5 = estimator.estimate_hand_penalty(attrs, turn=5)

    assert wr_t1 > 0.45, f"1-drop should be fine on turn 1: {wr_t1}"
    assert wr_t5 > 0.45, f"1-drop should be fine on turn 5: {wr_t5}"
    print(f"PASS: test_hand_penalty_cheap_card (t1={wr_t1:.3f}, t5={wr_t5:.3f})")


def test_resolver_resolve_with_turn_wr():
    from card_feature_resolver import CardFeatureResolver

    resolver = CardFeatureResolver()

    # Known card on board
    feats = resolver.resolve_with_turn_wr(turn=5, zone='board',
                                           card_name='lightning bolt')
    assert 'estimated_turn_wr' in feats
    assert 0.3 < feats['estimated_turn_wr'] < 0.8
    # Should still have regular features
    assert feats['is_instant'] == 1

    # Unknown card in hand (stuck)
    feats_stuck = resolver.resolve_with_turn_wr(
        turn=2, zone='hand', cmc=6, rarity='mythic', card_type='creature',
        power=6, toughness=6
    )
    assert feats_stuck['estimated_turn_wr'] < 0.5, "6-drop in hand on turn 2 should be penalized"

    # Same card on turn 7 (castable)
    feats_castable = resolver.resolve_with_turn_wr(
        turn=7, zone='hand', cmc=6, rarity='mythic', card_type='creature',
        power=6, toughness=6
    )
    assert feats_castable['estimated_turn_wr'] > feats_stuck['estimated_turn_wr']

    print("PASS: test_resolver_resolve_with_turn_wr")


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    test_estimator_loads()
    test_estimate_varies_by_turn()
    test_estimate_curve()
    test_hand_penalty_expensive_card()
    test_hand_penalty_cheap_card()
    test_resolver_resolve_with_turn_wr()

    print("\n=== ALL TURN-WR ESTIMATOR TESTS PASSED ===")
