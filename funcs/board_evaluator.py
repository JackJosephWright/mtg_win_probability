"""
Board state evaluation function for the MTG bot.

Two interfaces:
    1. evaluate(board_state) -> float   [0.0 = losing, 1.0 = winning]
    2. score_cards(turn, cards) -> dict  [card_name -> estimated value]

The bot calls evaluate() with hypothetical board states to rank candidate plays,
and score_cards() for fast pre-filtering of which cards are worth considering.

Usage:
    from board_evaluator import BoardEvaluator

    evaluator = BoardEvaluator()

    # Full board evaluation
    score = evaluator.evaluate({
        'turn': 5,
        'user_life': 18,
        'oppo_life': 14,
        'user_hand': ['Lightning Bolt', 'Mountain', 'Goblin Guide'],
        'user_creatures': ['Monastery Swiftspear'],
        'user_non_creatures': ['Experimental Frenzy'],
        'user_lands': 4,
        'oppo_creatures': ['Tarmogoyf'],
        'oppo_non_creatures': [],
        'oppo_lands': 3,
        'oppo_cards_in_hand': 2,
        'on_play': True,
    })
    # -> 0.71

    # Quick card scoring
    scores = evaluator.score_cards(5, ['Lightning Bolt', 'Tarmogoyf', 'Forest'])
    # -> {'Lightning Bolt': 0.54, 'Tarmogoyf': 0.61, 'Forest': 0.44}
"""
import os
import pickle
import time
import numpy as np
import pandas as pd

from card_feature_resolver import CardFeatureResolver
from card_attributes import FEATURE_NAMES, empty_features
from turn_wr_estimator import TurnWREstimator


# Zone aggregates computed for each card zone
ZONE_AGG_NAMES = [
    'count', 'total_power', 'total_toughness', 'total_cmc', 'total_loyalty',
    'avg_cmc', 'flyer_count', 'deathtouch_count', 'lifelink_count',
    'haste_count', 'hexproof_count', 'indestructible_count',
    'removal_count', 'etb_count', 'draw_count',
    'planeswalker_count', 'creature_count',
]


class BoardEvaluator:
    """
    Evaluates Magic board states using an XGBoost model trained on
    17Lands replay data with Scryfall card attributes.
    """

    def __init__(self, model_path=None, resolver=None):
        """
        Parameters:
            model_path: path to saved model pickle (default: models/board_eval/)
            resolver: CardFeatureResolver instance (will create one if None)
        """
        self.resolver = resolver or CardFeatureResolver()
        self._turn_wr = None

        # Load model
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'board_eval'
            )

        model_file = os.path.join(model_path, 'board_eval_xgb.pkl')
        cols_file = os.path.join(model_path, 'feature_columns.pkl')

        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(cols_file, 'rb') as f:
                self.feature_columns = pickle.load(f)
            print(f"BoardEvaluator: loaded model with {len(self.feature_columns)} features")
        else:
            self.model = None
            self.feature_columns = None
            print(f"BoardEvaluator: no model found at {model_file}, evaluate() unavailable")

    def evaluate(self, board_state):
        """
        Evaluate a board state and return win probability.

        Parameters:
            board_state: dict with keys:
                turn (int): current turn number
                user_life (float): player's life total
                oppo_life (float): opponent's life total
                user_hand (list[str]): card names in player's hand
                user_creatures (list[str]): card names of player's creatures
                user_non_creatures (list[str]): card names of player's non-creature permanents
                user_lands (int): number of player's lands
                oppo_creatures (list[str]): card names of opponent's creatures
                oppo_non_creatures (list[str]): card names of opponent's non-creature permanents
                oppo_lands (int): number of opponent's lands
                oppo_cards_in_hand (int): number of cards in opponent's hand
                on_play (bool): whether player went first

                Optional action features (default 0):
                user_total_mana_spent (float)
                oppo_total_mana_spent (float)
                user_total_combat_dmg_dealt (float)
                oppo_total_combat_dmg_dealt (float)
                user_total_creatures_killed (int)
                oppo_total_creatures_killed (int)
                user_total_creatures_lost (int)
                oppo_total_creatures_lost (int)
                user_creatures_attacked (int)
                oppo_creatures_attacked (int)

        Returns:
            float: win probability [0.0, 1.0]
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Train with train_board_eval.py first.")

        features = self._board_state_to_features(board_state)
        row = pd.DataFrame([features])

        # Align columns to training feature set
        for col in self.feature_columns:
            if col not in row.columns:
                row[col] = np.nan
        row = row[self.feature_columns]

        prob = self.model.predict_proba(row)[0, 1]
        return float(prob)

    def evaluate_batch(self, board_states):
        """Evaluate multiple board states at once. Returns list of win probabilities."""
        if self.model is None:
            raise RuntimeError("No model loaded.")

        rows = [self._board_state_to_features(bs) for bs in board_states]
        df = pd.DataFrame(rows)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_columns]

        probs = self.model.predict_proba(df)[:, 1]
        return [float(p) for p in probs]

    def score_cards(self, turn, card_names):
        """
        Quick card-level scoring for pre-filtering.

        Uses the turn-WR meta-model to estimate each card's value
        at the given turn. Context-free -- just card attributes + turn.

        Parameters:
            turn (int): current turn number
            card_names (list[str]): card names to score

        Returns:
            dict: {card_name: float} estimated win rate contribution
        """
        if self._turn_wr is None:
            try:
                self._turn_wr = TurnWREstimator()
            except FileNotFoundError:
                # No turn-WR model, use attribute-based heuristic
                return self._score_cards_heuristic(turn, card_names)

        scores = {}
        for name in card_names:
            feats = self.resolver.resolve(card_name=name)
            try:
                wr = self._turn_wr.estimate_from_features(feats, turn)
                scores[name] = float(wr)
            except Exception:
                scores[name] = 0.5  # neutral default
        return scores

    def _board_state_to_features(self, bs):
        """Convert a board_state dict to a flat feature dict matching training columns."""
        turn = bs.get('turn', 1)
        features = {}

        # Core features
        features['turn'] = turn
        features['on_play'] = int(bs.get('on_play', 0))
        features['user_life'] = bs.get('user_life', 20)
        features['oppo_life'] = bs.get('oppo_life', 20)
        features['user_lands_in_play'] = bs.get('user_lands', 0)
        features['oppo_lands_in_play'] = bs.get('oppo_lands', 0)
        features['oppo_cards_in_hand'] = bs.get('oppo_cards_in_hand', 0)

        # Derived
        features['life_diff'] = features['user_life'] - features['oppo_life']
        features['land_diff'] = features['user_lands_in_play'] - features['oppo_lands_in_play']

        # Action features
        features['user_total_mana_spent'] = bs.get('user_total_mana_spent', 0)
        features['oppo_total_mana_spent'] = bs.get('oppo_total_mana_spent', 0)
        features['user_total_combat_dmg_dealt'] = bs.get('user_total_combat_dmg_dealt', 0)
        features['oppo_total_combat_dmg_dealt'] = bs.get('oppo_total_combat_dmg_dealt', 0)
        features['user_total_combat_dmg_taken'] = bs.get('user_total_combat_dmg_taken', 0)
        features['oppo_total_combat_dmg_taken'] = bs.get('oppo_total_combat_dmg_taken', 0)
        features['user_total_creatures_killed'] = bs.get('user_total_creatures_killed', 0)
        features['oppo_total_creatures_killed'] = bs.get('oppo_total_creatures_killed', 0)
        features['user_total_creatures_lost'] = bs.get('user_total_creatures_lost', 0)
        features['oppo_total_creatures_lost'] = bs.get('oppo_total_creatures_lost', 0)
        features['user_creatures_attacked'] = bs.get('user_creatures_attacked', 0)
        features['oppo_creatures_attacked'] = bs.get('oppo_creatures_attacked', 0)

        # Card zones
        zones = {
            'user_hand': bs.get('user_hand', []),
            'user_creatures': bs.get('user_creatures', []),
            'user_non_creatures': bs.get('user_non_creatures', []),
            'oppo_creatures': bs.get('oppo_creatures', []),
            'oppo_non_creatures': bs.get('oppo_non_creatures', []),
        }

        for zone_name, cards in zones.items():
            zone_features = self._resolve_zone(zone_name, cards, 0)
            features.update(zone_features)

        return features

    def _resolve_zone(self, zone_prefix, card_names, max_cards):
        """Resolve a list of card names into zone aggregate features (no per-slot)."""
        features = {}
        resolved = []

        for name in card_names:
            if isinstance(name, dict):
                card_name = name.get('name', '')
                feats = self.resolver.resolve(card_name=card_name)
                if 'power' in name:
                    feats['power'] = float(name['power'])
                if 'toughness' in name:
                    feats['toughness'] = float(name['toughness'])
            else:
                feats = self.resolver.resolve(card_name=str(name))
            resolved.append(feats)

        count = len(resolved)
        features[f'{zone_prefix}_count'] = count

        if count == 0:
            features[f'{zone_prefix}_total_power'] = 0.0
            features[f'{zone_prefix}_total_toughness'] = 0.0
            features[f'{zone_prefix}_total_cmc'] = 0.0
            features[f'{zone_prefix}_total_loyalty'] = 0.0
            features[f'{zone_prefix}_avg_cmc'] = np.nan
            for k in ['flyer', 'deathtouch', 'lifelink', 'haste',
                       'hexproof', 'indestructible']:
                features[f'{zone_prefix}_{k}_count'] = 0
            for k in ['removal', 'etb', 'draw', 'planeswalker', 'creature']:
                features[f'{zone_prefix}_{k}_count'] = 0
            return features

        features[f'{zone_prefix}_total_power'] = sum(f.get('power', 0) or 0 for f in resolved)
        features[f'{zone_prefix}_total_toughness'] = sum(f.get('toughness', 0) or 0 for f in resolved)
        features[f'{zone_prefix}_total_cmc'] = sum(f.get('cmc', 0) or 0 for f in resolved)
        features[f'{zone_prefix}_total_loyalty'] = sum(f.get('loyalty', 0) or 0 for f in resolved)
        features[f'{zone_prefix}_avg_cmc'] = features[f'{zone_prefix}_total_cmc'] / count

        features[f'{zone_prefix}_flyer_count'] = sum(1 for f in resolved if f.get('kw_flying', 0) > 0.5)
        features[f'{zone_prefix}_deathtouch_count'] = sum(1 for f in resolved if f.get('kw_deathtouch', 0) > 0.5)
        features[f'{zone_prefix}_lifelink_count'] = sum(1 for f in resolved if f.get('kw_lifelink', 0) > 0.5)
        features[f'{zone_prefix}_haste_count'] = sum(1 for f in resolved if f.get('kw_haste', 0) > 0.5)
        features[f'{zone_prefix}_hexproof_count'] = sum(1 for f in resolved if f.get('kw_hexproof', 0) > 0.5)
        features[f'{zone_prefix}_indestructible_count'] = sum(1 for f in resolved if f.get('kw_indestructible', 0) > 0.5)
        features[f'{zone_prefix}_removal_count'] = sum(
            1 for f in resolved if (f.get('has_destroy', 0) > 0.5 or f.get('has_exile', 0) > 0.5))
        features[f'{zone_prefix}_etb_count'] = sum(1 for f in resolved if f.get('has_etb', 0) > 0.5)
        features[f'{zone_prefix}_draw_count'] = sum(1 for f in resolved if f.get('has_draw', 0) > 0.5)
        features[f'{zone_prefix}_planeswalker_count'] = sum(1 for f in resolved if f.get('is_planeswalker', 0) > 0.5)
        features[f'{zone_prefix}_creature_count'] = sum(1 for f in resolved if f.get('is_creature', 0) > 0.5)

        return features

    def _score_cards_heuristic(self, turn, card_names):
        """Fallback card scoring when turn-WR model isn't available."""
        scores = {}
        for name in card_names:
            feats = self.resolver.resolve(card_name=name)
            cmc = feats.get('cmc', 3) or 3
            power = feats.get('power', 0) or 0
            toughness = feats.get('toughness', 0) or 0
            rarity = feats.get('rarity_num', 1) or 1

            # Simple heuristic: higher stats and rarity = better
            # Penalize high CMC early, low impact late
            on_curve = 1.0 if abs(cmc - turn) <= 1 else 0.5
            stat_score = (power + toughness) / 10.0
            rarity_bonus = rarity * 0.02

            scores[name] = 0.45 + stat_score * 0.1 + rarity_bonus + on_curve * 0.03
            scores[name] = max(0.3, min(0.7, scores[name]))

        return scores
