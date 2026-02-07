"""
Extracts a fixed-length numeric feature vector from any Magic card.

Works for ALL card types (creatures, planeswalkers, instants, sorceries,
artifacts, enchantments, tokens) and ALL cards ever printed.

Feature vector (~35 features):
    - Mana: cmc, color_count, W/U/B/R/G bits, is_colorless
    - Stats: power, toughness, loyalty
    - Type: is_creature, is_planeswalker, is_instant, is_sorcery,
            is_artifact, is_enchantment, is_legendary
    - Keywords: flying, trample, deathtouch, lifelink, haste, vigilance,
               reach, menace, first_strike, double_strike, flash,
               hexproof, indestructible, defender, ward
    - Oracle text signals: has_draw, has_destroy, has_exile, has_counter,
                          has_damage, has_etb, has_activated, has_token_gen,
                          oracle_word_count
    - Rarity: rarity_num (1-4)

For unseen cards: if the bot provides CMC + type + rarity + colors,
we can still build a meaningful feature vector even without oracle text.
"""
import re
import numpy as np

RARITY_MAP = {
    'common': 1, 'uncommon': 2, 'rare': 3, 'mythic': 4, 'bonus': 2, 'special': 3,
    'C': 1, 'U': 2, 'R': 3, 'M': 4, 'B': 1,
}

# Ordered list of keywords to check
KEYWORD_LIST = [
    'flying', 'trample', 'deathtouch', 'lifelink', 'haste', 'vigilance',
    'reach', 'menace', 'first strike', 'double strike', 'flash',
    'hexproof', 'indestructible', 'defender', 'ward',
]

# Feature names in order (for DataFrame column headers)
FEATURE_NAMES = (
    ['cmc', 'power', 'toughness', 'loyalty', 'color_count',
     'is_white', 'is_blue', 'is_black', 'is_red', 'is_green', 'is_colorless',
     'is_creature', 'is_planeswalker', 'is_instant', 'is_sorcery',
     'is_artifact', 'is_enchantment', 'is_legendary',
     'rarity_num']
    + [f'kw_{kw.replace(" ", "_")}' for kw in KEYWORD_LIST]
    + ['has_draw', 'has_destroy', 'has_exile', 'has_counter',
       'has_damage', 'has_etb', 'has_activated', 'has_token_gen',
       'oracle_word_count']
)

NUM_FEATURES = len(FEATURE_NAMES)


def card_to_features(card_attrs):
    """
    Convert a card attribute dict (from Scryfall) to a numeric feature vector.

    Parameters:
        card_attrs: dict with keys from scryfall_loader (name, cmc, type_line,
                    oracle_text, power, toughness, loyalty, colors,
                    color_identity, keywords, rarity)

    Returns:
        dict: {feature_name: float_value} for all features
    """
    cmc = min(float(card_attrs.get('cmc', 0) or 0), 20.0)  # cap at 20
    power = float(card_attrs.get('power', 0) or 0)
    toughness = float(card_attrs.get('toughness', 0) or 0)
    loyalty = float(card_attrs.get('loyalty', 0) or 0)

    # Colors
    colors = set(card_attrs.get('colors', []) or card_attrs.get('color_identity', []))
    color_count = len(colors)
    is_colorless = 1 if color_count == 0 else 0

    # Type line parsing
    type_line = (card_attrs.get('type_line', '') or '').lower()

    # Keywords (from Scryfall keywords list + oracle text fallback)
    keywords_list = [k.lower() for k in (card_attrs.get('keywords', []) or [])]
    oracle = (card_attrs.get('oracle_text', '') or '').lower()
    # Also check oracle text for keywords (Scryfall sometimes misses them)
    all_kw_text = ' '.join(keywords_list) + ' ' + oracle

    # Rarity
    rarity = card_attrs.get('rarity', 'common')
    rarity_num = RARITY_MAP.get(rarity, RARITY_MAP.get(str(rarity).lower(), 1))

    feats = {
        # Mana
        'cmc': cmc,
        'power': power,
        'toughness': toughness,
        'loyalty': loyalty,
        'color_count': color_count,
        'is_white': 1 if 'W' in colors else 0,
        'is_blue': 1 if 'U' in colors else 0,
        'is_black': 1 if 'B' in colors else 0,
        'is_red': 1 if 'R' in colors else 0,
        'is_green': 1 if 'G' in colors else 0,
        'is_colorless': is_colorless,

        # Types
        'is_creature': 1 if 'creature' in type_line else 0,
        'is_planeswalker': 1 if 'planeswalker' in type_line else 0,
        'is_instant': 1 if 'instant' in type_line else 0,
        'is_sorcery': 1 if 'sorcery' in type_line else 0,
        'is_artifact': 1 if 'artifact' in type_line else 0,
        'is_enchantment': 1 if 'enchantment' in type_line else 0,
        'is_legendary': 1 if 'legendary' in type_line else 0,

        # Rarity
        'rarity_num': rarity_num,
    }

    # Keywords
    for kw in KEYWORD_LIST:
        feat_name = f'kw_{kw.replace(" ", "_")}'
        feats[feat_name] = 1 if kw in all_kw_text else 0

    # Oracle text signals
    feats['has_draw'] = 1 if re.search(r'draw[s]?\s+(a\s+)?card', oracle) else 0
    feats['has_destroy'] = 1 if 'destroy' in oracle else 0
    feats['has_exile'] = 1 if 'exile' in oracle else 0
    feats['has_counter'] = 1 if re.search(r'counter\s+target', oracle) else 0
    feats['has_damage'] = 1 if re.search(r'deal[s]?\s+\d+\s+damage', oracle) else 0
    feats['has_etb'] = 1 if ('enters the battlefield' in oracle or 'enters,' in oracle or 'enters.' in oracle) else 0
    feats['has_activated'] = 1 if ':' in oracle else 0
    feats['has_token_gen'] = 1 if re.search(r'create[s]?\s+(a|an|\d+|x)\s+', oracle) else 0
    feats['oracle_word_count'] = len(oracle.split()) if oracle else 0

    return feats


def card_to_feature_vector(card_attrs):
    """Returns features as a numpy array in FEATURE_NAMES order."""
    feats = card_to_features(card_attrs)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)


def empty_features():
    """Returns NaN feature dict for an empty card slot."""
    return {name: np.nan for name in FEATURE_NAMES}


def minimal_card_attrs(cmc=None, rarity=None, card_type=None, colors=None,
                       power=None, toughness=None):
    """
    Build a minimal card_attrs dict from whatever the bot knows.

    This is the fallback for completely unseen cards where the bot can
    provide basic info from the card object (e.g., Arena API, Scryfall search).

    Example:
        attrs = minimal_card_attrs(cmc=4, rarity='rare', card_type='creature',
                                    colors=['B'], power=4, toughness=5)
        features = card_to_features(attrs)
    """
    type_map = {
        'creature': 'Creature',
        'instant': 'Instant',
        'sorcery': 'Sorcery',
        'planeswalker': 'Planeswalker',
        'artifact': 'Artifact',
        'enchantment': 'Enchantment',
        'land': 'Land',
    }

    type_line = type_map.get(card_type, card_type or '')

    return {
        'cmc': cmc or 0,
        'type_line': type_line,
        'oracle_text': '',
        'power': power or 0,
        'toughness': toughness or 0,
        'loyalty': 0,
        'colors': colors or [],
        'color_identity': colors or [],
        'keywords': [],
        'rarity': rarity or 'common',
    }
