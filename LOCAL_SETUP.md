# Board Evaluator - Local Setup

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download replay data from 17Lands
Download replay data for multiple sets. More sets = better model.

```bash
mkdir -p data/raw_csv
cd data/raw_csv

# Download these (each ~400-600MB compressed):
curl -LO "https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.DSK.PremierDraft.csv.gz"
curl -LO "https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.FDN.PremierDraft.csv.gz"
curl -LO "https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.BLB.PremierDraft.csv.gz"
curl -LO "https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.OTJ.PremierDraft.csv.gz"
curl -LO "https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.MKM.PremierDraft.csv.gz"
curl -LO "https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.WOE.PremierDraft.csv.gz"
curl -LO "https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.LCI.PremierDraft.csv.gz"

cd ../..
```

### 3. Process replay data into training features
This transforms raw replay data into turn-level rows with zone aggregate features.
Uses a card resolution cache so it only does Scryfall lookup once per unique card.

```bash
python -u main/process_multi_set.py --games-per-set 5000 --max-turns 15
```

**Expected output**: `data/processed_csv/multi_set_turns.parquet`
- ~7 sets x 5000 drafts x 6 games x 7 turns = ~1.5M turn rows
- ~120 feature columns (zone aggregates + action features + scalars)
- Takes ~15-30 min depending on CPU, needs ~4-8GB RAM

**Tune the parameters:**
- `--games-per-set 10000` for more data (better model, slower)
- `--games-per-set 1000` for a quick test run

### 4. Train the XGBoost board evaluation model
```bash
python main/train_board_eval.py --input data/processed_csv/multi_set_turns.parquet
```

**Expected output**: `models/board_eval/board_eval_xgb.pkl`
- Trains XGBoost with game-level split (no data leakage)
- Prints AUC, Brier score, per-turn metrics, feature importances
- Target: AUC > 0.75, Brier < 0.20

### 5. Use the BoardEvaluator in your bot
```python
import sys
sys.path.insert(0, 'funcs')
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
print(f"Win probability: {score:.1%}")

# Quick card scoring (turn-WR meta-model, no board context)
scores = evaluator.score_cards(5, ['Lightning Bolt', 'Tarmogoyf', 'Forest'])
print(scores)

# Batch evaluation (faster for comparing multiple hypothetical states)
states = [state_after_play_A, state_after_play_B, state_after_play_C]
probs = evaluator.evaluate_batch(states)
best = max(zip(probs, ['A', 'B', 'C']))
```

## Architecture

```
17Lands replay data (game-per-row, ~1M games per set)
    |
    v
process_multi_set.py  -- samples games, resolves cards via Scryfall, extracts zone aggregates
    |
    v
multi_set_turns.parquet  -- turn-per-row, ~120 features
    |
    v
train_board_eval.py  -- XGBoost with game-level split
    |
    v
board_eval_xgb.pkl  -- saved model
    |
    v
BoardEvaluator.evaluate(board_state) -> win probability
```

## Feature Types

**Zone aggregates (17 per zone x 5 zones = 85):**
count, total_power, total_toughness, total_cmc, total_loyalty, avg_cmc,
flyer_count, deathtouch_count, lifelink_count, haste_count, hexproof_count,
indestructible_count, removal_count, etb_count, draw_count,
planeswalker_count, creature_count

**Scalar features (~10):**
turn, on_play, user_life, oppo_life, user_lands_in_play, oppo_lands_in_play,
oppo_cards_in_hand, life_diff, land_diff

**Action features (~12):**
user/oppo_total_mana_spent, total_combat_dmg_dealt, total_combat_dmg_taken,
total_creatures_killed, total_creatures_lost, creatures_attacked

**Card scorer (separate model):**
turn-WR meta-model predicts card value from Scryfall attributes + turn number
