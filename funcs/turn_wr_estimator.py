"""
Estimates turn-specific win rate for any card based on its attributes.

Uses a meta-model trained on 58k (card_attrs, turn) -> WR observations from
17Lands Limited data. Generalizes to unseen cards: given a card's CMC, P/T,
type, keywords, and rarity, predicts what its WR would be on any given turn.

Key patterns learned:
    - Cheap creatures: flat WR curve, slight decay late
    - Expensive bombs: high early (if on board), gradual decay
    - Flyers/deathtouch: WR increases relative to vanilla over time
    - Removal: flat curve (always relevant)
    - Planeswalkers: high and relatively stable

Limitations:
    - Trained on battlefield-state WR (creatures/non-creatures in play)
    - Doesn't directly capture in-hand dynamics (expensive card stuck in hand)
    - For in-hand estimation, use cmc_minus_turn as a penalty signal
    - R²=0.39 on held-out cards -- captures broad patterns, not card-specific nuance
"""
import os
import numpy as np
import joblib

from card_attributes import card_to_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'turn_wr_estimator')

# Features the meta-model expects (must match training)
META_FEATURE_COLS = [
    'turn',
    'cmc', 'power', 'toughness', 'loyalty',
    'is_creature', 'is_instant', 'is_sorcery', 'is_planeswalker',
    'is_artifact', 'is_enchantment',
    'rarity_num',
    'has_flying', 'has_deathtouch', 'has_lifelink', 'has_trample', 'has_haste',
    'color_count', 'oracle_word_count',
    'turn_cmc_ratio', 'cmc_minus_turn', 'on_curve',
]


class TurnWREstimator:
    """
    Predicts turn-specific win rate for any card from its attributes.

    Usage:
        estimator = TurnWREstimator()

        # From Scryfall attributes dict
        card_attrs = scryfall_db['lightning bolt']
        wr_turn_5 = estimator.estimate(card_attrs, turn=5)

        # From card_to_features output
        feats = card_to_features(card_attrs)
        wr_turn_5 = estimator.estimate_from_features(feats, turn=5)

        # Get full curve (turns 1-15)
        curve = estimator.estimate_curve(card_attrs)
        # -> {1: 0.52, 2: 0.54, 3: 0.53, ...}
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, 'turn_wr_xgb.pkl')
        self.model = joblib.load(model_path)

    def estimate(self, card_attrs, turn):
        """
        Estimate WR for a card (Scryfall attrs dict) at a given turn.

        Parameters:
            card_attrs: dict with Scryfall fields (cmc, type_line, oracle_text, etc.)
            turn: int, the turn number

        Returns:
            float: estimated win rate at that turn
        """
        feats = card_to_features(card_attrs)
        return self.estimate_from_features(feats, turn)

    def estimate_from_features(self, feats, turn):
        """
        Estimate WR from a pre-computed feature dict + turn number.

        Parameters:
            feats: dict from card_to_features()
            turn: int

        Returns:
            float: estimated win rate
        """
        cmc = feats.get('cmc', 0) or 0
        row = self._build_meta_row(feats, turn, cmc)
        return float(self.model.predict([row])[0])

    def estimate_curve(self, card_attrs, max_turn=15):
        """
        Estimate WR curve across all turns for a card.

        Returns:
            dict: {turn: estimated_wr} for turns 1 through max_turn
        """
        feats = card_to_features(card_attrs)
        cmc = feats.get('cmc', 0) or 0

        curve = {}
        rows = [self._build_meta_row(feats, t, cmc) for t in range(1, max_turn + 1)]
        preds = self.model.predict(rows)
        for t, pred in zip(range(1, max_turn + 1), preds):
            curve[t] = float(pred)
        return curve

    def estimate_hand_penalty(self, card_attrs, turn):
        """
        Estimate a penalty for holding an expensive card in hand.

        Cards with CMC > turn are stuck in hand and represent dead draws.
        Returns a value < 0.5 (penalty) or > 0.5 (playable) to use as
        a feature in the win probability model.

        This is a simple heuristic since we don't have in-hand WR training data.
        """
        cmc = float(card_attrs.get('cmc', 0) or 0)
        if cmc <= turn:
            # Card is playable -- use the normal board estimate
            return self.estimate(card_attrs, turn)
        else:
            # Card is stuck in hand -- penalize based on how far from castable
            turns_until_castable = cmc - turn
            # Baseline 0.5 (neutral), penalize ~0.02 per turn until castable
            penalty = 0.5 - (turns_until_castable * 0.02)
            return max(0.35, penalty)  # floor at 0.35

    def _build_meta_row(self, feats, turn, cmc):
        """Build feature row for the meta-model."""
        cmc = max(cmc, 1)  # avoid div by zero
        kw_map = {
            'has_flying': 'kw_flying',
            'has_deathtouch': 'kw_deathtouch',
            'has_lifelink': 'kw_lifelink',
            'has_trample': 'kw_trample',
            'has_haste': 'kw_haste',
        }
        row = []
        for col in META_FEATURE_COLS:
            if col == 'turn':
                row.append(turn)
            elif col == 'turn_cmc_ratio':
                row.append(turn / cmc)
            elif col == 'cmc_minus_turn':
                row.append(feats.get('cmc', 0) - turn)
            elif col == 'on_curve':
                card_cmc = feats.get('cmc', 0) or 0
                row.append(1 if card_cmc <= turn <= card_cmc + 1 else 0)
            elif col in kw_map:
                row.append(feats.get(kw_map[col], 0))
            else:
                row.append(feats.get(col, 0))
        return row
