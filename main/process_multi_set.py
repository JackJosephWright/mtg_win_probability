"""
Process multiple 17Lands replay data sets into a unified turn-level training dataset.

Uses ZONE AGGREGATES ONLY (no per-slot card features) to keep memory manageable.
This is also better for the model -- no arbitrary slot ordering, and aggregates
capture the board state more compactly.

Features per zone (5 zones): count, total_power, total_toughness, total_cmc,
    total_loyalty, avg_cmc, flyer_count, deathtouch_count, lifelink_count,
    haste_count, hexproof_count, indestructible_count, removal_count,
    etb_count, draw_count, planeswalker_count, creature_count

Usage:
    python process_multi_set.py --games-per-set 3000 --max-turns 15
"""
import os
import sys
import argparse
import glob
import gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'funcs'))

from card_feature_resolver import CardFeatureResolver
from card_attributes import FEATURE_NAMES, empty_features

META_COLS = [
    'expansion', 'draft_id', 'on_play', 'num_turns', 'won',
]

EOT_CARD_ZONES = {
    'user_hand': ('user', 'eot_user_cards_in_hand'),
    'user_creatures': ('user', 'eot_user_creatures_in_play'),
    'user_non_creatures': ('user', 'eot_user_non_creatures_in_play'),
    'oppo_creatures': ('oppo', 'eot_oppo_creatures_in_play'),
    'oppo_non_creatures': ('oppo', 'eot_oppo_non_creatures_in_play'),
}

_EMPTY_ARR = np.array([empty_features()[f] for f in FEATURE_NAMES], dtype=np.float32)

# Feature indices for aggregate computation
_IDX = {f: FEATURE_NAMES.index(f) for f in FEATURE_NAMES}


class CachedResolver:
    def __init__(self, resolver):
        self.resolver = resolver
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def resolve_id(self, card_id_str):
        if card_id_str in self._cache:
            self.hits += 1
            return self._cache[card_id_str]
        self.misses += 1
        feats = self.resolver.resolve(card_id=card_id_str)
        arr = np.array([feats[f] for f in FEATURE_NAMES], dtype=np.float32)
        self._cache[card_id_str] = arr
        return arr

    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total > 0 else 0
        return f"Cache: {len(self._cache)} unique, {rate:.0f}% hit ({self.hits}/{total})"


def _count_pipe(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    s = str(val).strip()
    if not s or s == 'nan':
        return 0
    return len(s.split('|'))


def _count_lands(val):
    s = str(val)
    if s == 'nan' or not s.strip():
        return 0
    return s.count('|') + 1


def _resolve_zone_aggregates(pipe_str, cached, prefix):
    """Resolve pipe-delimited card IDs and compute zone aggregates directly."""
    result = {}

    if not isinstance(pipe_str, str) or not pipe_str.strip() or pipe_str == 'nan':
        result[f'{prefix}_count'] = 0
        result[f'{prefix}_total_power'] = 0.0
        result[f'{prefix}_total_toughness'] = 0.0
        result[f'{prefix}_total_cmc'] = 0.0
        result[f'{prefix}_total_loyalty'] = 0.0
        result[f'{prefix}_avg_cmc'] = np.nan
        result[f'{prefix}_flyer_count'] = 0
        result[f'{prefix}_deathtouch_count'] = 0
        result[f'{prefix}_lifelink_count'] = 0
        result[f'{prefix}_haste_count'] = 0
        result[f'{prefix}_hexproof_count'] = 0
        result[f'{prefix}_indestructible_count'] = 0
        result[f'{prefix}_removal_count'] = 0
        result[f'{prefix}_etb_count'] = 0
        result[f'{prefix}_draw_count'] = 0
        result[f'{prefix}_planeswalker_count'] = 0
        result[f'{prefix}_creature_count'] = 0
        return result

    ids = pipe_str.split('|')
    count = len(ids)

    # Resolve all cards and stack
    arrays = []
    for cid in ids:
        cid = cid.split('.')[0].strip()
        if cid:
            arrays.append(cached.resolve_id(cid))

    if not arrays:
        result[f'{prefix}_count'] = 0
        for k in ['total_power', 'total_toughness', 'total_cmc', 'total_loyalty']:
            result[f'{prefix}_{k}'] = 0.0
        result[f'{prefix}_avg_cmc'] = np.nan
        for k in ['flyer_count', 'deathtouch_count', 'lifelink_count', 'haste_count',
                   'hexproof_count', 'indestructible_count', 'removal_count',
                   'etb_count', 'draw_count', 'planeswalker_count', 'creature_count']:
            result[f'{prefix}_{k}'] = 0
        return result

    stacked = np.stack(arrays)
    count = len(arrays)

    result[f'{prefix}_count'] = count
    result[f'{prefix}_total_power'] = float(np.nansum(stacked[:, _IDX['power']]))
    result[f'{prefix}_total_toughness'] = float(np.nansum(stacked[:, _IDX['toughness']]))
    result[f'{prefix}_total_cmc'] = float(np.nansum(stacked[:, _IDX['cmc']]))
    result[f'{prefix}_total_loyalty'] = float(np.nansum(stacked[:, _IDX['loyalty']]))
    result[f'{prefix}_avg_cmc'] = result[f'{prefix}_total_cmc'] / count

    for kw, feat in [('flyer', 'kw_flying'), ('deathtouch', 'kw_deathtouch'),
                     ('lifelink', 'kw_lifelink'), ('haste', 'kw_haste'),
                     ('hexproof', 'kw_hexproof'), ('indestructible', 'kw_indestructible')]:
        result[f'{prefix}_{kw}_count'] = int(np.sum(stacked[:, _IDX[feat]] > 0.5))

    result[f'{prefix}_removal_count'] = (
        int(np.sum(stacked[:, _IDX['has_destroy']] > 0.5)) +
        int(np.sum(stacked[:, _IDX['has_exile']] > 0.5)))
    result[f'{prefix}_etb_count'] = int(np.sum(stacked[:, _IDX['has_etb']] > 0.5))
    result[f'{prefix}_draw_count'] = int(np.sum(stacked[:, _IDX['has_draw']] > 0.5))
    result[f'{prefix}_planeswalker_count'] = int(np.sum(stacked[:, _IDX['is_planeswalker']] > 0.5))
    result[f'{prefix}_creature_count'] = int(np.sum(stacked[:, _IDX['is_creature']] > 0.5))

    return result


def transform_game(row, cached, max_turns=15):
    """Transform one game row into list of turn dicts with aggregate features only."""
    num_turns = min(int(row['num_turns']), max_turns)
    game_id = row['draft_id']
    turns = []

    for turn in range(1, num_turns + 1):
        t = {
            'game_id': game_id,
            'expansion': row.get('expansion', ''),
            'turn': turn,
            'on_play': int(row.get('on_play', 0)),
            'won': int(row.get('won', 0)),
        }

        # Card zone aggregates
        for zone_prefix, (player, suffix) in EOT_CARD_ZONES.items():
            col = f'{player}_turn_{turn}_{suffix}'
            pipe_str = row.get(col, None)
            t.update(_resolve_zone_aggregates(pipe_str, cached, zone_prefix))

        # Life totals
        for p in ['user', 'oppo']:
            t[f'{p}_life'] = row.get(f'{p}_turn_{turn}_eot_{p}_life', np.nan)
            t[f'{p}_lands_in_play'] = _count_lands(
                row.get(f'{p}_turn_{turn}_eot_{p}_lands_in_play', None))

        t['oppo_cards_in_hand'] = row.get(
            f'oppo_turn_{turn}_eot_oppo_cards_in_hand', np.nan)

        # Cumulative action features
        for p in ['user', 'oppo']:
            tm, tdd, tdt, tk, tl = 0.0, 0.0, 0.0, 0, 0
            for ti in range(1, turn + 1):
                m = row.get(f'{p}_turn_{ti}_{p}_mana_spent', 0)
                tm += float(m) if not pd.isna(m) else 0
                dd = row.get(f'{p}_turn_{ti}_oppo_combat_damage_taken', 0)
                tdd += float(dd) if not pd.isna(dd) else 0
                dt = row.get(f'{p}_turn_{ti}_user_combat_damage_taken', 0)
                tdt += float(dt) if not pd.isna(dt) else 0
                tk += _count_pipe(row.get(f'{p}_turn_{ti}_oppo_creatures_killed_combat', None))
                tk += _count_pipe(row.get(f'{p}_turn_{ti}_oppo_creatures_killed_non_combat', None))
                tl += _count_pipe(row.get(f'{p}_turn_{ti}_user_creatures_killed_combat', None))
                tl += _count_pipe(row.get(f'{p}_turn_{ti}_user_creatures_killed_non_combat', None))

            t[f'{p}_total_mana_spent'] = tm
            t[f'{p}_total_combat_dmg_dealt'] = tdd
            t[f'{p}_total_combat_dmg_taken'] = tdt
            t[f'{p}_total_creatures_killed'] = tk
            t[f'{p}_total_creatures_lost'] = tl
            t[f'{p}_creatures_attacked'] = _count_pipe(
                row.get(f'{p}_turn_{turn}_creatures_attacked', None))

        # Derived
        ul = t.get('user_life', 20) or 20
        ol = t.get('oppo_life', 20) or 20
        t['life_diff'] = ul - ol
        t['land_diff'] = (t.get('user_lands_in_play', 0) or 0) - (t.get('oppo_lands_in_play', 0) or 0)

        turns.append(t)
    return turns


def get_needed_columns(max_turns=15):
    cols = list(META_COLS)
    action_suffixes = [
        'user_mana_spent', 'oppo_mana_spent',
        'oppo_combat_damage_taken', 'user_combat_damage_taken',
        'oppo_creatures_killed_combat', 'user_creatures_killed_combat',
        'oppo_creatures_killed_non_combat', 'user_creatures_killed_non_combat',
        'creatures_attacked',
    ]
    eot_suffixes = [
        'eot_user_cards_in_hand', 'eot_oppo_cards_in_hand',
        'eot_user_lands_in_play', 'eot_oppo_lands_in_play',
        'eot_user_creatures_in_play', 'eot_oppo_creatures_in_play',
        'eot_user_non_creatures_in_play', 'eot_oppo_non_creatures_in_play',
        'eot_user_life', 'eot_oppo_life',
    ]
    for turn in range(1, max_turns + 1):
        for p in ['user', 'oppo']:
            for s in eot_suffixes:
                cols.append(f'{p}_turn_{turn}_{s}')
            for s in action_suffixes:
                cols.append(f'{p}_turn_{turn}_{s}')
    return cols


def process_set(filepath, cached, games_per_set, max_turns, output_dir):
    set_code = Path(filepath).stem.split('.')[1]
    print(f"\n{'='*60}", flush=True)
    print(f"Processing {set_code} ...", flush=True)

    needed = get_needed_columns(max_turns)
    header_cols = pd.read_csv(filepath, nrows=0).columns.tolist()
    use_cols = [c for c in needed if c in header_cols]

    read_rows = games_per_set * 8
    print(f"  Reading {read_rows:,} rows, {len(use_cols)} cols ...", flush=True)

    try:
        raw = pd.read_csv(filepath, nrows=read_rows, usecols=use_cols, low_memory=False)
    except Exception:
        raw = pd.read_csv(filepath, nrows=read_rows, low_memory=False)
        raw = raw[[c for c in use_cols if c in raw.columns]]

    unique_drafts = raw['draft_id'].unique()
    rng = np.random.RandomState(42)
    n = min(games_per_set, len(unique_drafts))
    sampled = set(rng.choice(unique_drafts, size=n, replace=False))
    data = raw[raw['draft_id'].isin(sampled)]
    del raw; gc.collect()
    print(f"  {n:,} drafts -> {len(data):,} games", flush=True)

    all_turns = []
    for i, (_, row) in enumerate(data.iterrows()):
        all_turns.extend(transform_game(row, cached, max_turns))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,}/{len(data):,} games -> {len(all_turns):,} turns | {cached.stats()}", flush=True)

    print(f"  Done: {len(data):,} games -> {len(all_turns):,} turns | {cached.stats()}", flush=True)

    out = output_dir / f'turns_{set_code}.parquet'
    df = pd.DataFrame(all_turns)
    df.to_parquet(str(out), index=False)
    print(f"  Saved {out.name} ({out.stat().st_size / 1e6:.1f} MB, {len(df.columns)} cols)", flush=True)

    del all_turns, df, data; gc.collect()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-per-set', type=int, default=10000)
    parser.add_argument('--max-turns', type=int, default=15)
    parser.add_argument('--output', type=str, default='data/processed_csv/multi_set_turns.parquet')
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'raw_csv'
    files = sorted(glob.glob(str(data_dir / 'replay_data_public.*.PremierDraft.csv.gz')))
    print(f"Found {len(files)} sets", flush=True)

    resolver = CardFeatureResolver()
    cached = CachedResolver(resolver)

    output_path = Path(__file__).resolve().parent.parent / args.output
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pq_files = []
    for f in files:
        pq_files.append(process_set(f, cached, args.games_per_set, args.max_turns, output_dir))

    print(f"\nCombining {len(pq_files)} files ...", flush=True)
    combined = pd.concat([pd.read_parquet(str(p)) for p in pq_files], ignore_index=True)
    print(f"Combined: {len(combined):,} turns, {combined['game_id'].nunique():,} games, {len(combined.columns)} cols", flush=True)
    print(f"Sets: {combined['expansion'].value_counts().to_dict()}", flush=True)

    combined.to_parquet(str(output_path), index=False)
    sz = output_path.stat().st_size / 1e6
    print(f"Saved {output_path} ({sz:.1f} MB)", flush=True)

    for p in pq_files:
        p.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
