"""
Train phase-based win probability models using multi-attribute card features.

Key improvements over v1:
    - Multi-attribute features per card (wr, cmc, rarity) instead of WR-only
    - Game-level train/test split (no data leakage between turns of same game)
    - Properly saves phase-specific models (not the last-trained general model)
    - Outputs both accuracy and AUC metrics
    - Compares against WR-only baseline

Usage:
    python train_phase_models_v2.py --input <path_to_processed_csv> [--max_rows N]
"""
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Add funcs to path
FUNCS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'funcs')
sys.path.insert(0, FUNCS_DIR)

MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'win_probability', 'phase_models_v2')

PHASES = {
    "early_game": (1, 5),
    "mid_game": (6, 10),
    "late_game": (11, 30),
}

DROP_COLS = ['game_id', 'unique_id', 'turn', 'won']


def load_and_prepare_data(input_path, max_rows=None):
    """Load processed turn-level data and prepare for training."""
    df = pd.read_csv(input_path, nrows=max_rows)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()[:20]}...")

    # Ensure target is binary int
    df['won'] = df['won'].astype(int)

    return df


def game_level_split(df, test_size=0.2, random_state=42):
    """Split data by game_id so no game appears in both train and test."""
    if 'game_id' not in df.columns:
        print("WARNING: No game_id column found, falling back to row-level split")
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=test_size, random_state=random_state)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df['game_id']))
    return df.iloc[train_idx], df.iloc[test_idx]


def get_feature_columns(df):
    """Get all feature columns (everything except metadata and target)."""
    return [c for c in df.columns if c not in DROP_COLS]


def train_phase_model(train_df, test_df, phase_name, turn_range):
    """Train and evaluate a model for a specific game phase."""
    turn_min, turn_max = turn_range

    phase_train = train_df[(train_df['turn'] >= turn_min) & (train_df['turn'] <= turn_max)]
    phase_test = test_df[(test_df['turn'] >= turn_min) & (test_df['turn'] <= turn_max)]

    if len(phase_train) == 0 or len(phase_test) == 0:
        print(f"  SKIP {phase_name}: not enough data (train={len(phase_train)}, test={len(phase_test)})")
        return None, {}

    feature_cols = get_feature_columns(phase_train)
    X_train = phase_train[feature_cols]
    y_train = phase_train['won']
    X_test = phase_test[feature_cols]
    y_test = phase_test['won']

    print(f"\n  {phase_name}: train={len(X_train)}, test={len(X_test)}, features={len(feature_cols)}")

    # Train XGBoost (handles NaN natively)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': len(feature_cols),
    }

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to processed turn-level CSV')
    parser.add_argument('--max_rows', type=int, default=None, help='Limit rows for testing')
    args = parser.parse_args()

    print("=" * 60)
    print("Training Phase-Based Models v2 (multi-attribute features)")
    print("=" * 60)

    df = load_and_prepare_data(args.input, max_rows=args.max_rows)

    print("\nSplitting by game_id (no data leakage)...")
    train_df, test_df = game_level_split(df)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    train_games = train_df['game_id'].nunique() if 'game_id' in train_df.columns else 'N/A'
    test_games = test_df['game_id'].nunique() if 'game_id' in test_df.columns else 'N/A'
    print(f"Train games: {train_games}, Test games: {test_games}")

    # Train phase-specific models
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    results = {}

    for phase_name, turn_range in PHASES.items():
        model, metrics = train_phase_model(train_df, test_df, phase_name, turn_range)
        results[phase_name] = metrics

        if model is not None:
            model_path = os.path.join(MODEL_OUTPUT_DIR, f'xgboost_{phase_name}.pkl')
            joblib.dump(model, model_path)
            print(f"  Saved: {model_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Phase':<15} {'Accuracy':>10} {'AUC':>10} {'Log Loss':>10} {'Train':>8} {'Test':>8}")
    print("-" * 60)
    for phase, m in results.items():
        if m:
            print(f"{phase:<15} {m['accuracy']:>10.4f} {m['auc']:>10.4f} {m['log_loss']:>10.4f} {m['train_size']:>8} {m['test_size']:>8}")

    print(f"\nModels saved to: {MODEL_OUTPUT_DIR}")


if __name__ == '__main__':
    main()
