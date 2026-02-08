# Board Evaluator - Local Setup

## 3-Step Pipeline

```bash
pip install -r requirements.txt

# Step 1: Download 17Lands replay data (~3.5GB, 7 sets)
python main/download_data.py

# Step 2: Process into training features (~15-30 min)
python -u main/process_multi_set.py --games-per-set 10000

# Step 3: Train XGBoost model (~1-2 min)
python main/train_board_eval.py
```

That's it. You'll have a trained model at `models/board_eval/board_eval_xgb.pkl`.

## Use in Bot

```python
import sys
sys.path.insert(0, 'funcs')
from board_evaluator import BoardEvaluator

evaluator = BoardEvaluator()

# Full board evaluation -> win probability [0.0, 1.0]
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
print(f"Win probability: {score:.1%}")

# Quick card scoring (context-free, fast pre-filter)
scores = evaluator.score_cards(5, ['Lightning Bolt', 'Tarmogoyf', 'Forest'])

# Batch evaluation (compare hypothetical plays)
probs = evaluator.evaluate_batch([state_A, state_B, state_C])
```

## Options

**Download more/fewer sets:**
```bash
python main/download_data.py --list               # see all available sets
python main/download_data.py --sets DSK FDN BLB   # specific sets
python main/download_data.py --all                 # every available set
```

**Process more/fewer games:**
```bash
python -u main/process_multi_set.py --games-per-set 20000  # more data, better model
python -u main/process_multi_set.py --games-per-set 1000   # quick test
```

## Architecture

```
17Lands replay data (game-per-row, ~1M games/set)
    |  download_data.py
    v
data/raw_csv/*.csv.gz
    |  process_multi_set.py (card resolution cache, zone aggregates)
    v
data/processed_csv/multi_set_turns.parquet (~120 features/row)
    |  train_board_eval.py (XGBoost, game-level split)
    v
models/board_eval/board_eval_xgb.pkl
    |
    v
BoardEvaluator.evaluate(board_state) -> win probability
BoardEvaluator.score_cards(turn, cards) -> card scores
```

## Features (~120 total)

**Zone aggregates (17 features x 5 zones = 85):**
Cards are resolved via Scryfall DB to attributes, then aggregated per zone.
Zones: user_hand, user_creatures, user_non_creatures, oppo_creatures, oppo_non_creatures

Per zone: count, total_power, total_toughness, total_cmc, total_loyalty, avg_cmc,
flyer_count, deathtouch_count, lifelink_count, haste_count, hexproof_count,
indestructible_count, removal_count, etb_count, draw_count,
planeswalker_count, creature_count

**Scalar features (~10):**
turn, on_play, user_life, oppo_life, user_lands, oppo_lands,
oppo_cards_in_hand, life_diff, land_diff

**Action features (~12):**
Cumulative per game: mana_spent, combat_dmg_dealt, combat_dmg_taken,
creatures_killed, creatures_lost, creatures_attacked (for both players)

**Card scorer** (separate turn-WR meta-model):
Predicts card value from Scryfall attributes + turn number.
Used for fast pre-filtering before full board evaluation.
