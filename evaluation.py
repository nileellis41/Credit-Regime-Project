# evaluation.py
"""
Side-by-side evaluation and visualization of HMM vs LSTM credit regimes.

Outputs
-------
  results/regime_comparison.png   — main 4-panel comparison chart
  results/confusion_matrices.png  — LSTM test confusion + HMM agreement
  results/regime_stats.csv        — per-regime macro statistics
  results/metrics_summary.txt     — printed comparison table
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    cohen_kappa_score, accuracy_score
)
from scipy.stats import entropy
from config import (
    N_REGIMES, REGIME_NAMES, REGIME_COLORS, RESULTS_DIR, SEQUENCE_LEN
)


os.makedirs(RESULTS_DIR, exist_ok=True)

PALETTE = [REGIME_COLORS[i] for i in range(N_REGIMES)]


# ─── Regime statistics ────────────────────────────────────────────────────────

def regime_stats(
    states: np.ndarray,
    dates: pd.DatetimeIndex,
    df_raw: pd.DataFrame,
    model_name: str = "Model",
) -> pd.DataFrame:
    """
    For each regime: mean HY OAS, mean unrate, mean ci_tighten, frequency, avg duration.
    """
    df = df_raw.reindex(dates).copy()
    df["regime"] = states

    rows = []
    for r in range(N_REGIMES):
        mask = df["regime"] == r
        n_months = mask.sum()
        if n_months == 0:
            continue

        # Average duration of continuous spells
        transitions = np.diff(mask.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        if mask.iloc[0]:
            starts = np.concatenate([[0], starts])
        ends = np.where(transitions == -1)[0] + 1
        if mask.iloc[-1]:
            ends = np.concatenate([ends, [len(mask)]])
        durations = ends - starts if len(starts) == len(ends) else [n_months]
        avg_dur = np.mean(durations) if len(durations) else n_months

        row = {
            "model": model_name,
            "regime": REGIME_NAMES[r],
            "freq_%": round(100 * n_months / len(df), 1),
            "avg_duration_months": round(avg_dur, 1),
            "mean_hy_oas": round(df.loc[mask, "hy_oas"].mean(), 0) if "hy_oas" in df else np.nan,
            "mean_unrate": round(df.loc[mask, "unrate"].mean(), 2) if "unrate" in df else np.nan,
            "mean_ci_tighten": round(df.loc[mask, "ci_tighten"].mean(), 1) if "ci_tighten" in df else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ─── Agreement between models ─────────────────────────────────────────────────

def model_agreement(
    hmm_states: np.ndarray,
    lstm_states: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict:
    """
    Agreement metrics between HMM and LSTM on the full time series.
    """
    acc = accuracy_score(hmm_states, lstm_states)
    kappa = cohen_kappa_score(hmm_states, lstm_states)
    cm = confusion_matrix(hmm_states, lstm_states, labels=list(range(N_REGIMES)))

    # Regime-specific agreement
    per_regime = {}
    for r in range(N_REGIMES):
        mask = hmm_states == r
        if mask.sum() > 0:
            per_regime[REGIME_NAMES[r]] = round(
                accuracy_score(hmm_states[mask], lstm_states[mask]), 3
            )

    return {
        "overall_agreement": round(acc, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": cm,
        "per_regime_agreement": per_regime,
    }


# ─── Transition timing ────────────────────────────────────────────────────────

def transition_lag(
    hmm_states: np.ndarray,
    lstm_states: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict:
    """
    For each LSTM transition (t → t+1), find the nearest HMM transition.
    Positive lag = LSTM leads HMM; negative = LSTM lags.
    """
    def get_transition_dates(states):
        changes = np.where(np.diff(states) != 0)[0] + 1
        return changes  # indices

    hmm_trans  = set(get_transition_dates(hmm_states))
    lstm_trans = set(get_transition_dates(lstm_states))

    if not lstm_trans:
        return {"mean_lag_months": np.nan, "n_lstm_transitions": 0}

    lags = []
    for lt in lstm_trans:
        if hmm_trans:
            nearest_hmm = min(hmm_trans, key=lambda h: abs(h - lt))
            lags.append(lt - nearest_hmm)

    return {
        "mean_lag_months": round(float(np.mean(lags)), 2),
        "std_lag_months":  round(float(np.std(lags)), 2),
        "n_lstm_transitions": len(lstm_trans),
        "n_hmm_transitions":  len(hmm_trans),
    }


# ─── Posterior entropy (regime uncertainty) ───────────────────────────────────

def posterior_entropy_comparison(
    hmm_post: np.ndarray,
    lstm_post: np.ndarray,
) -> dict:
    """Mean Shannon entropy of posterior distributions — lower = more confident."""
    hmm_ent  = np.mean([entropy(p + 1e-9) for p in hmm_post])
    lstm_ent = np.mean([entropy(p + 1e-9) for p in lstm_post])
    return {
        "hmm_mean_entropy":  round(hmm_ent, 4),
        "lstm_mean_entropy": round(lstm_ent, 4),
        "more_confident":    "HMM" if hmm_ent < lstm_ent else "LSTM",
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def _shade_regimes(ax, states, dates, alpha=0.25):
    """Shade background of ax by regime color."""
    current = states[0]
    start   = dates[0]
    for i in range(1, len(states)):
        if states[i] != current or i == len(states) - 1:
            end = dates[i]
            ax.axvspan(start, end, alpha=alpha,
                       color=REGIME_COLORS[current], linewidth=0)
            current = states[i]
            start = dates[i]


def plot_regime_comparison(
    dates: pd.DatetimeIndex,
    hmm_states: np.ndarray,
    lstm_states: np.ndarray,
    hmm_post: np.ndarray,
    lstm_post: np.ndarray,
    df_raw: pd.DataFrame,
    save_path: str = None,
):
    """4-panel figure: HY OAS + regimes, regime posteriors (HMM), regime posteriors (LSTM), agreement."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
    fig.suptitle("Credit Regime Detector: LSTM vs HMM Baseline", fontsize=16, fontweight="bold", y=0.98)

    df_plot = df_raw.reindex(dates)

    # Panel 1: HY OAS with regime shading (HMM)
    ax = axes[0]
    ax.plot(dates, df_plot["hy_oas"], color="#2c3e50", linewidth=1.2, label="HY OAS")
    if "ig_oas" in df_plot:
        ax.plot(dates, df_plot["ig_oas"], color="#7f8c8d", linewidth=0.9, linestyle="--", label="IG OAS")
    _shade_regimes(ax, hmm_states, dates)
    ax.set_ylabel("OAS (bp)", fontsize=10)
    ax.set_title("Credit Spreads — HMM Regime Background", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: HMM posterior probabilities
    ax = axes[1]
    for r in range(N_REGIMES):
        ax.fill_between(dates, hmm_post[:, r], alpha=0.6,
                        color=REGIME_COLORS[r], label=REGIME_NAMES[r])
    ax.set_ylabel("Posterior P(regime)", fontsize=10)
    ax.set_title("HMM — Smoothed State Posteriors", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Panel 3: LSTM posterior probabilities
    ax = axes[2]
    for r in range(N_REGIMES):
        ax.fill_between(dates, lstm_post[:, r], alpha=0.6,
                        color=REGIME_COLORS[r], label=REGIME_NAMES[r])
    ax.set_ylabel("Posterior P(regime)", fontsize=10)
    ax.set_title("LSTM — Softmax State Posteriors", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Panel 4: Rolling 12m agreement between HMM and LSTM
    ax = axes[3]
    agree = (hmm_states == lstm_states).astype(float)
    rolling_agree = pd.Series(agree, index=dates).rolling(12).mean()
    ax.plot(dates, rolling_agree, color="#8e44ad", linewidth=1.3)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, label="70% threshold")
    ax.fill_between(dates, rolling_agree, 0, alpha=0.15, color="#8e44ad")
    ax.set_ylabel("Agreement (12m rolling)", fontsize=10)
    ax.set_title("HMM vs LSTM — Rolling Agreement", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Legend patches
    patches = [mpatches.Patch(color=REGIME_COLORS[r], label=REGIME_NAMES[r]) for r in range(N_REGIMES)]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10, title="Regimes",
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = save_path or os.path.join(RESULTS_DIR, "regime_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[eval] Saved → {save_path}")
    plt.close()


def plot_confusion_and_agreement(
    lstm_test_preds: np.ndarray,
    lstm_test_labels: np.ndarray,
    hmm_states: np.ndarray,
    lstm_states_full: np.ndarray,
    save_path: str = None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("LSTM Test Confusion & HMM-LSTM Agreement Matrix", fontsize=14, fontweight="bold")

    labels = list(range(N_REGIMES))
    label_names = [REGIME_NAMES[r] for r in labels]

    # LSTM test confusion matrix
    cm_lstm = confusion_matrix(lstm_test_labels, lstm_test_preds, labels=labels)
    cm_lstm_norm = cm_lstm.astype(float) / cm_lstm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_lstm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                ax=ax1, vmin=0, vmax=1)
    ax1.set_title("LSTM Test Set — Normalized Confusion\n(True=HMM Labels)", fontsize=11)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")

    # HMM vs LSTM agreement confusion
    cm_agree = confusion_matrix(hmm_states, lstm_states_full, labels=labels)
    cm_agree_norm = cm_agree.astype(float) / cm_agree.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_agree_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=label_names, yticklabels=label_names,
                ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Full Series — HMM (rows) vs LSTM (cols)\nNormalized Agreement", fontsize=11)
    ax2.set_xlabel("LSTM Regime"); ax2.set_ylabel("HMM Regime")

    plt.tight_layout()
    save_path = save_path or os.path.join(RESULTS_DIR, "confusion_matrices.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[eval] Saved → {save_path}")
    plt.close()


# ─── Summary report ───────────────────────────────────────────────────────────

def print_summary(
    hmm_metrics: dict,
    lstm_history: dict,
    agreement: dict,
    lag: dict,
    entropy_cmp: dict,
    hmm_stats: pd.DataFrame,
    lstm_stats: pd.DataFrame,
    lstm_test_preds: np.ndarray,
    lstm_test_labels: np.ndarray,
):
    sep = "=" * 70
    lines = [
        sep,
        "  CREDIT REGIME DETECTOR — MODEL COMPARISON REPORT",
        sep,
        "",
        "── HMM BASELINE ─────────────────────────────────────────────────────",
        f"  Log-likelihood : {hmm_metrics['log_likelihood']}",
        f"  AIC            : {hmm_metrics['aic']}",
        f"  BIC            : {hmm_metrics['bic']}",
        f"  Converged      : {hmm_metrics['converged']}",
        "",
        "── LSTM CLASSIFIER ──────────────────────────────────────────────────",
        f"  Test accuracy  : {lstm_history['test_acc']:.4f}",
        f"  (vs HMM pseudo-labels — measures replication fidelity)",
        "",
        classification_report(
            lstm_test_labels, lstm_test_preds,
            target_names=[REGIME_NAMES[r] for r in range(N_REGIMES)],
            zero_division=0,
        ),
        "",
        "── MODEL AGREEMENT (full time series) ───────────────────────────────",
        f"  Overall agreement : {agreement['overall_agreement']}",
        f"  Cohen's kappa     : {agreement['cohens_kappa']}",
        f"  Per-regime        : {agreement['per_regime_agreement']}",
        "",
        "── TRANSITION TIMING ────────────────────────────────────────────────",
        f"  LSTM vs HMM mean lag : {lag['mean_lag_months']} months",
        f"  (positive = LSTM leads, negative = LSTM lags)",
        f"  LSTM transitions : {lag['n_lstm_transitions']}",
        f"  HMM  transitions : {lag['n_hmm_transitions']}",
        "",
        "── POSTERIOR CONFIDENCE ─────────────────────────────────────────────",
        f"  HMM  mean entropy  : {entropy_cmp['hmm_mean_entropy']}",
        f"  LSTM mean entropy  : {entropy_cmp['lstm_mean_entropy']}",
        f"  More confident     : {entropy_cmp['more_confident']}",
        "",
        "── REGIME STATISTICS ────────────────────────────────────────────────",
        "  HMM regimes:",
        hmm_stats.to_string(index=False),
        "",
        "  LSTM regimes:",
        lstm_stats.to_string(index=False),
        sep,
    ]
    report = "\n".join(lines)
    print(report)
    path = os.path.join(RESULTS_DIR, "metrics_summary.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"[eval] Report saved → {path}")
