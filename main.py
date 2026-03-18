# main.py
"""
LSTM Credit Regime Detector — Main entry point.

Usage
-----
  python main.py              # Full train + eval pipeline
  python main.py --no-cache   # Force-refresh FRED data
  python main.py --load       # Load saved models, skip training
  python main.py --help

Pipeline
--------
  1. Fetch FRED data (with caching)
  2. Build feature matrix
  3. Fit GaussianHMM (3 states) → Viterbi labels
  4. Train LSTM classifier on HMM labels
  5. Run full-series inference for both models
  6. Compare: agreement, timing lag, entropy, regime stats
  7. Write charts + report to results/
"""

import argparse
import os
import numpy as np
import pandas as pd

from config import (
    N_REGIMES, REGIME_NAMES, MODEL_DIR, RESULTS_DIR,
    SEQUENCE_LEN, RANDOM_SEED,
)
from data_fetcher import fetch_raw, refresh_cache
from features import build_features
from hmm_model import (
    fit_hmm, hmm_predict, hmm_metrics, save_hmm, load_hmm,
    regime_transition_matrix,
)
from lstm_model import (
    train_lstm, lstm_predict_full, save_lstm, load_lstm,
)
from evaluation import (
    regime_stats, model_agreement, transition_lag,
    posterior_entropy_comparison, plot_regime_comparison,
    plot_confusion_and_agreement, print_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Credit Regime Detector")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-fetch FRED data")
    parser.add_argument("--load", action="store_true",
                        help="Load pre-trained models from models/ dir")
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LEN,
                        help=f"LSTM sequence length (default: {SEQUENCE_LEN})")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 — FRED DATA")
    print("=" * 60)
    if args.no_cache:
        df_raw = refresh_cache()
    else:
        df_raw = fetch_raw(use_cache=True)

    # ── 2. Features ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 — FEATURE ENGINEERING")
    print("=" * 60)
    X, dates, feat_names, scaler = build_features(df_raw)
    n_features = X.shape[1]
    print(f"  Features ({n_features}): {feat_names}")

    # ── 3. HMM ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 — GAUSSIAN HMM BASELINE")
    print("=" * 60)
    if args.load and os.path.exists(os.path.join(MODEL_DIR, "hmm.pkl")):
        hmm = load_hmm()
    else:
        hmm = fit_hmm(X, dates)
        save_hmm(hmm)

    hmm_states, hmm_post = hmm_predict(hmm, X, dates, align=True)
    h_metrics = hmm_metrics(hmm, X)

    print("\n  Empirical Transition Matrix (HMM):")
    print(regime_transition_matrix(hmm_states).to_string())

    hmm_counts = pd.Series(hmm_states).value_counts().sort_index()
    print("\n  HMM State Counts:")
    for r, cnt in hmm_counts.items():
        print(f"    {REGIME_NAMES[r]:10s}: {cnt} months ({100*cnt/len(hmm_states):.1f}%)")

    # ── 4. LSTM ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 — LSTM CLASSIFIER")
    print("=" * 60)
    seq_len = args.seq_len

    if args.load and os.path.exists(os.path.join(MODEL_DIR, "lstm.pt")):
        lstm_model = load_lstm(input_size=n_features)
        # We still need history for evaluation — run a quick eval pass
        lstm_model, history = train_lstm(X, hmm_states, seq_len=seq_len)
    else:
        lstm_model, history = train_lstm(X, hmm_states, seq_len=seq_len)
        save_lstm(lstm_model)

    # ── 5. Full-series inference ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5 — FULL-SERIES INFERENCE")
    print("=" * 60)
    lstm_states, lstm_post = lstm_predict_full(lstm_model, X, seq_len=seq_len)

    lstm_counts = pd.Series(lstm_states).value_counts().sort_index()
    print("\n  LSTM State Counts:")
    for r, cnt in lstm_counts.items():
        print(f"    {REGIME_NAMES[r]:10s}: {cnt} months ({100*cnt/len(lstm_states):.1f}%)")

    # ── 6. Evaluation ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6 — EVALUATION & COMPARISON")
    print("=" * 60)
    df_raw_aligned = df_raw.reindex(dates)

    agreement   = model_agreement(hmm_states, lstm_states, dates)
    lag         = transition_lag(hmm_states, lstm_states, dates)
    entropy_cmp = posterior_entropy_comparison(hmm_post, lstm_post)
    hmm_stats   = regime_stats(hmm_states,  dates, df_raw_aligned, "HMM")
    lstm_stats  = regime_stats(lstm_states, dates, df_raw_aligned, "LSTM")

    # Combine and save regime stats
    all_stats = pd.concat([hmm_stats, lstm_stats])
    all_stats.to_csv(os.path.join(RESULTS_DIR, "regime_stats.csv"), index=False)
    print(f"[eval] Regime stats → {RESULTS_DIR}/regime_stats.csv")

    # Print and save text report
    print_summary(
        hmm_metrics=h_metrics,
        lstm_history=history,
        agreement=agreement,
        lag=lag,
        entropy_cmp=entropy_cmp,
        hmm_stats=hmm_stats,
        lstm_stats=lstm_stats,
        lstm_test_preds=history["test_preds"],
        lstm_test_labels=history["test_labels"],
    )

    # ── 7. Charts ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7 — CHARTS")
    print("=" * 60)
    plot_regime_comparison(
        dates=dates,
        hmm_states=hmm_states,
        lstm_states=lstm_states,
        hmm_post=hmm_post,
        lstm_post=lstm_post,
        df_raw=df_raw_aligned,
    )
    plot_confusion_and_agreement(
        lstm_test_preds=history["test_preds"],
        lstm_test_labels=history["test_labels"],
        hmm_states=hmm_states,
        lstm_states_full=lstm_states,
    )

    print("\n✓  Pipeline complete. Outputs in results/")
    print(f"   regime_comparison.png")
    print(f"   confusion_matrices.png")
    print(f"   regime_stats.csv")
    print(f"   metrics_summary.txt")


if __name__ == "__main__":
    main()
