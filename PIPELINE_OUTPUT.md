# Pipeline Output (Board Evaluator)

## What we did

Ran the full board-evaluation pipeline (`main/process_multi_set.py` -> `main/train_board_eval.py`) across 12 MTG draft sets. This:

1. **Downloaded** raw game CSVs via `main/download_data.py`
2. **Transformed** each game into per-turn board-state snapshots with card attribute features (CMC, rarity, keywords, type line, etc.)
3. **Trained** an XGBoost model to predict win probability from any board state

## Generated files (gitignored, local only)

| File | Size | Description |
|------|------|-------------|
| `data/processed_csv/multi_set_turns.parquet` | ~164 MB | Combined turn-level training data (6M rows, 12 sets, 104 features) |
| `data/processed_csv/turns_*.parquet` | varies | Per-set intermediate parquet files |
| `models/board_eval/board_eval_xgb.pkl` | — | Trained XGBoost model |
| `models/board_eval/feature_columns.pkl` | — | Feature column list used by the model |
| `models/board_eval/training_summary.json` | — | Training metrics (see below) |

These files are in `.gitignore` because they're too large for git / are reproducible.

## How to reproduce

```bash
# 1. Download raw data
python main/download_data.py

# 2. Process into turn-level parquet
python main/process_multi_set.py

# 3. Train the model
python main/train_board_eval.py
```

## Training results

- **Sets**: STX, SNC, MKM, MOM, LCI, WOE, OTJ, FDN, DMU, DSK, BLB, AFR
- **Data**: 5,993,014 turns from 119,997 games
- **Features**: 104
- **Test accuracy**: 71.2%
- **Test AUC**: 0.794
- **Test Brier score**: 0.182
- **Train time**: ~92s

## Bug fixes applied

The latest commit (`11ab82c`) fixed runtime issues found during pipeline execution:

- **`card_attributes.py`**: NaN handling for `type_line`, `keywords`, `oracle_text` (some cards have null fields)
- **`process_multi_set.py`**: Dynamic column detection (`draft_id`/`history_id`, `num_turns`/`turns`), safe float conversion, numeric coercion for mixed-type columns
