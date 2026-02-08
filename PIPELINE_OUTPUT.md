# Pipeline Output (Board Evaluator)

## What we did

Ran the full board-evaluation pipeline across all 12 17Lands draft sets:

1. **Downloaded** 12 raw game CSVs via `main/download_data.py` (STX, SNC, MKM, MOM, LCI, WOE, OTJ, FDN, DMU, DSK, BLB, AFR)
2. **Processed each set** into per-turn board-state snapshots with card attribute features (CMC, rarity, keywords, type line, etc.) via `main/process_multi_set.py`
3. **Combined all 12 sets** into a single unified dataset: `data/processed_csv/multi_set_turns.parquet` (~164 MB, 5.99M rows across 119,997 games, 104 features)
4. **Trained** an XGBoost model on the combined dataset via `main/train_board_eval.py`

## Generated files (gitignored — too large for GitHub, must be reproduced locally)

All output files exceed GitHub's 100 MB limit or are reproducible artifacts, so they are **not pushed to GitHub**. They live only on your local machine after running the pipeline.

| File | Size | Description |
|------|------|-------------|
| `data/raw_csv/replay_data_public.*.csv.gz` | ~1–2 GB each | Raw 17Lands game data (12 sets) |
| `data/processed_csv/multi_set_turns.parquet` | ~164 MB | **The combined 12-set training dataset** (6M turn-level rows, 104 features) |
| `models/board_eval/board_eval_xgb.pkl` | — | Trained XGBoost model |
| `models/board_eval/feature_columns.pkl` | — | Feature column list used by the model |
| `models/board_eval/training_summary.json` | — | Training metrics (see below) |

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
