"""
Resolves any card (by name, ID, or minimal attributes) to a full feature vector.

Resolution hierarchy:
    1. Card name in Scryfall DB -> full attribute features (oracle text, keywords, P/T, etc.)
    2. Card ID in 17Lands cards.csv -> partial features (name, CMC, rarity, types)
       then try name lookup in Scryfall DB
    3. Bot provides minimal attrs (CMC, type, rarity, colors) -> synthetic features
    4. Nothing known -> global average features

This is the main interface for the bot. Given any card identifier, it returns
a consistent feature vector usable by the win probability model.
"""
import os
import json
import numpy as np
import pandas as pd

from card_attributes import (
    card_to_features, empty_features, minimal_card_attrs,
    FEATURE_NAMES, RARITY_MAP,
)


class CardFeatureResolver:
    """
    Resolves any card to a numeric feature vector.

    Initialize once, then call resolve() for each card.
    """

    def __init__(self, scryfall_db_path=None, cards_csv_path=None):
        """
        Parameters:
            scryfall_db_path: path to processed Scryfall JSON from scryfall_loader.py
                              (optional, enables full feature extraction)
            cards_csv_path: path to existing cards.csv from 17Lands
                            (optional, enables ID->name lookup)
        """
        self.scryfall_db = {}
        self.id_to_name = {}
        self.id_to_attrs = {}
        self._avg_features = None

        # Load Scryfall DB if available
        if scryfall_db_path is None:
            scryfall_db_path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'scryfall', 'card_attributes.json'
            )
        if os.path.exists(scryfall_db_path):
            with open(scryfall_db_path, 'r') as f:
                self.scryfall_db = json.load(f)
            print(f"CardFeatureResolver: loaded {len(self.scryfall_db)} cards from Scryfall DB")

        # Load 17Lands cards.csv for ID->name mapping
        if cards_csv_path is None:
            cards_csv_path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'cards_data', 'cards.csv'
            )
        if os.path.exists(cards_csv_path):
            cards_df = pd.read_csv(cards_csv_path)
            for _, row in cards_df.iterrows():
                card_id = int(row['id'])
                name = row['name']
                self.id_to_name[card_id] = name
                self.id_to_attrs[card_id] = {
                    'name': name,
                    'cmc': row.get('mana_value', 0),
                    'rarity': row.get('rarity', 'common'),
                    'type_line': row.get('types', ''),
                    'color_identity': list(str(row.get('color_identity', '') or '')),
                }
            print(f"CardFeatureResolver: loaded {len(self.id_to_name)} card IDs from cards.csv")

    def resolve(self, card_id=None, card_name=None, **minimal_kwargs):
        """
        Resolve a card to its feature dict.

        Provide ONE of:
            - card_id: 17Lands numeric ID (int or str)
            - card_name: exact card name (str)
            - minimal_kwargs: cmc, rarity, card_type, colors, power, toughness

        Returns:
            dict: {feature_name: float} with keys from FEATURE_NAMES
        """
        # Try by name in Scryfall DB (most complete)
        if card_name is not None:
            result = self._resolve_by_name(card_name)
            if result is not None:
                return result

        # Try by ID -> get name -> try Scryfall
        if card_id is not None:
            result = self._resolve_by_id(card_id)
            if result is not None:
                return result

        # Minimal attributes provided by bot
        if minimal_kwargs:
            attrs = minimal_card_attrs(**minimal_kwargs)
            return card_to_features(attrs)

        # Global average fallback
        return self._get_avg_features()

    def resolve_nan(self):
        """Returns NaN features for an empty card slot."""
        return empty_features()

    def _resolve_by_name(self, name):
        """Look up card by name in Scryfall DB."""
        key = name.lower().strip()
        if key in self.scryfall_db:
            return card_to_features(self.scryfall_db[key])
        # Try front face of double-faced cards (e.g., "Fable of the Mirror-Breaker")
        for db_key, attrs in self.scryfall_db.items():
            if db_key.startswith(key) or key in db_key:
                return card_to_features(attrs)
        return None

    def _resolve_by_id(self, card_id):
        """Look up card by 17Lands ID, then try Scryfall by name."""
        try:
            card_id_int = int(float(str(card_id).split('.')[0]))
        except (ValueError, TypeError):
            return None

        if card_id_int not in self.id_to_name:
            return None

        name = self.id_to_name[card_id_int]

        # Try full Scryfall lookup by name
        result = self._resolve_by_name(name)
        if result is not None:
            return result

        # Fall back to partial attrs from cards.csv
        partial = self.id_to_attrs[card_id_int]
        return card_to_features({
            'name': partial['name'],
            'cmc': partial.get('cmc', 0),
            'type_line': partial.get('type_line', ''),
            'oracle_text': '',  # not available from cards.csv
            'power': 0,
            'toughness': 0,
            'loyalty': 0,
            'colors': partial.get('color_identity', []),
            'color_identity': partial.get('color_identity', []),
            'keywords': [],
            'rarity': partial.get('rarity', 'common'),
        })

    def _get_avg_features(self):
        """Compute global average features from Scryfall DB (cached)."""
        if self._avg_features is not None:
            return self._avg_features

        if not self.scryfall_db:
            # No Scryfall data - return generic defaults
            self._avg_features = card_to_features(minimal_card_attrs(
                cmc=3, rarity='common', card_type='creature', power=2, toughness=2
            ))
            return self._avg_features

        # Average across all cards in DB
        all_feats = [card_to_features(attrs) for attrs in self.scryfall_db.values()]
        avg = {}
        for feat_name in FEATURE_NAMES:
            vals = [f[feat_name] for f in all_feats if not np.isnan(f[feat_name])]
            avg[feat_name] = np.mean(vals) if vals else 0.0

        self._avg_features = avg
        return self._avg_features
