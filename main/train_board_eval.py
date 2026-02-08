"""
Train the board evaluation XGBoost model on multi-set turn-level data.

Reads processed parquet, does game-level train/test split, trains XGBoost,
evaluates metrics, and saves the model for BoardEvaluator.

Usage:
    python train_board_eval.py [--input data/processed_csv/multi_set_turns.parquet]
"""
import os
import sys
import argparse
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, log_loss
)

# Add funcs to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'funcs'))


def get_feature_columns(df):
    """Identify feature columns (everything except meta/target)."""
    exclude = {'game_id', 'expansion', 'won', 'unique_id'}
    return [c for c in df.columns if c not in exclude]


def train_and_evaluate(input_path, output_dir):
    """Train XGBoost board evaluator and save model."""
    print(f"Loading data from {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Sets: {df['expansion'].value_counts().to_dict()}")
    print(f"  Win rate: {df['won'].mean():.3f}")
    print(f"  Games: {df['game_id'].nunique():,}")

    # Feature columns
    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)}")

    X = df[feature_cols]
    y = df['won'].astype(int)
    groups = df['game_id']

    # Game-level split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"\n  Train: {len(X_train):,} rows ({groups.iloc[train_idx].nunique():,} games)")
    print(f"  Test:  {len(X_test):,} rows ({groups.iloc[test_idx].nunique():,} games)")

    # Train XGBoost
    try:
        from xgboost import XGBClassifier
        print("\nTraining XGBoost ...")
        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("\nXGBoost not available, using HistGradientBoosting ...")
        model = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42,
        )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"TEST METRICS (game-level split, {len(X_test):,} rows)")
    print(f"{'='*50}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  AUC:        {auc:.4f}")
    print(f"  Brier:      {brier:.4f}")
    print(f"  Log Loss:   {ll:.4f}")

    # Per-turn breakdown
    test_df = df.iloc[test_idx].copy()
    test_df['y_prob'] = y_prob
    test_df['y_true'] = y_test.values

    print(f"\nPer-turn breakdown:")
    print(f"  {'Turn':>5} {'Count':>8} {'AUC':>7} {'Brier':>7} {'Acc':>7}")
    for turn in sorted(test_df['turn'].unique()):
        mask = test_df['turn'] == turn
        if mask.sum() < 50:
            continue
        t_prob = test_df.loc[mask, 'y_prob']
        t_true = test_df.loc[mask, 'y_true']
        if t_true.nunique() < 2:
            continue
        t_auc = roc_auc_score(t_true, t_prob)
        t_brier = brier_score_loss(t_true, t_prob)
        t_acc = accuracy_score(t_true, (t_prob >= 0.5).astype(int))
        print(f"  {turn:>5} {mask.sum():>8,} {t_auc:>7.3f} {t_brier:>7.4f} {t_acc:>7.3f}")

    # Calibration table
    print(f"\nCalibration (predicted vs actual):")
    print(f"  {'Bucket':>12} {'Count':>8} {'Pred':>7} {'Actual':>7} {'Gap':>7}")
    for lo in np.arange(0, 1, 0.1):
        hi = lo + 0.1
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() < 10:
            continue
        pred_mean = y_prob[mask].mean()
        actual_mean = y_test.values[mask].mean()
        gap = actual_mean - pred_mean
        print(f"  [{lo:.1f}, {hi:.1f}) {mask.sum():>8,} {pred_mean:>7.3f} {actual_mean:>7.3f} {gap:>+7.3f}")

    # Feature importances (top 30)
    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(feature_cols, model.feature_importances_))
        sorted_imp = sorted(importances.items(), key=lambda x: -x[1])[:30]
        print(f"\nTop 30 feature importances:")
        for name, imp in sorted_imp:
            print(f"  {name:>50} {imp:.4f}")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'board_eval_xgb.pkl')
    cols_path = os.path.join(output_dir, 'feature_columns.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)

    # Save training summary
    import json
    summary = {
        'total_rows': len(df),
        'total_games': int(df['game_id'].nunique()),
        'sets': df['expansion'].value_counts().to_dict(),
        'n_features': len(feature_cols),
        'test_accuracy': round(acc, 4),
        'test_auc': round(auc, 4),
        'test_brier': round(brier, 4),
        'test_log_loss': round(ll, 4),
        'train_time_seconds': round(train_time, 1),
    }
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Feature columns saved to {cols_path}")

    return model, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='data/processed_csv/multi_set_turns.parquet')
    parser.add_argument('--output-dir', type=str,
                        default='models/board_eval')
    args = parser.parse_args()

    input_path = Path(__file__).resolve().parent.parent / args.input
    output_dir = Path(__file__).resolve().parent.parent / args.output_dir

    train_and_evaluate(str(input_path), str(output_dir))


if __name__ == '__main__':
    main()
