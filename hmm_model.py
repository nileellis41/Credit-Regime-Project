# hmm_model.py
"""
GaussianHMM baseline — 3-state regime detector.

The HMM is trained on the full feature set (same as LSTM).
After fitting, states are semantically aligned using known stress anchors
so that: 0=Benign, 1=Stress, 2=Crisis.
"""

import os
import pickle
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from config import (
    N_REGIMES, HMM_N_ITER, HMM_COV_TYPE,
    STRESS_ANCHORS, MODEL_DIR, REGIME_NAMES,
)


# ─── Fit ──────────────────────────────────────────────────────────────────────

def fit_hmm(X: np.ndarray, dates: pd.DatetimeIndex) -> GaussianHMM:
    """
    Fit a Gaussian HMM with N_REGIMES states.
    Uses multiple random restarts and keeps the best log-likelihood.
    """
    best_model, best_ll = None, -np.inf

    for seed in range(10):
        model = GaussianHMM(
            n_components=N_REGIMES,
            covariance_type=HMM_COV_TYPE,
            n_iter=HMM_N_ITER,
            random_state=seed,
            verbose=False,
        )
        try:
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll, best_model = ll, model
        except Exception:
            continue

    print(f"[HMM] Best log-likelihood: {best_ll:.2f}  (converged: {best_model.monitor_.converged})")
    return best_model


# ─── State alignment ──────────────────────────────────────────────────────────

def _align_states(
    model: GaussianHMM,
    raw_states: np.ndarray,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Map raw HMM state integers to semantic labels {0=Benign, 1=Stress, 2=Crisis}
    using STRESS_ANCHORS.

    Strategy
    --------
    1. For each anchor (date, expected_label), record which raw state was active.
    2. Build a vote matrix: vote_matrix[raw_state, expected_label] += 1
    3. Solve the assignment via argmax; fall back to spread-mean ordering if votes tie.
    """
    date_to_idx = {d.strftime("%Y-%m"): i for i, d in enumerate(dates)}
    vote_matrix = np.zeros((N_REGIMES, N_REGIMES), dtype=int)

    for anchor_date, expected_label in STRESS_ANCHORS:
        if anchor_date in date_to_idx:
            raw = raw_states[date_to_idx[anchor_date]]
            vote_matrix[raw, expected_label] += 1

    # Greedy assignment: raw_state → semantic_label
    mapping = {}
    used_labels = set()
    for _ in range(N_REGIMES):
        r, c = np.unravel_index(vote_matrix.argmax(), vote_matrix.shape)
        if vote_matrix[r, c] == 0:
            break
        mapping[r] = c
        used_labels.add(c)
        vote_matrix[r, :] = -1
        vote_matrix[:, c] = -1

    # Fill any unmapped states with remaining labels
    remaining = [l for l in range(N_REGIMES) if l not in used_labels]
    for raw in range(N_REGIMES):
        if raw not in mapping:
            mapping[raw] = remaining.pop(0)

    aligned = np.array([mapping[s] for s in raw_states], dtype=np.int64)
    print(f"[HMM] State mapping (raw → semantic): {mapping}")
    return aligned


# ─── Predict ──────────────────────────────────────────────────────────────────

def hmm_predict(
    model: GaussianHMM,
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    align: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Viterbi decode + posterior probabilities.

    Returns
    -------
    states  : (T,)  aligned integer regime labels
    posteriors : (T, N_REGIMES)  smoothed state probabilities
    """
    raw_states = model.predict(X)
    posteriors = model.predict_proba(X)

    if align:
        aligned = _align_states(model, raw_states, dates)
        # Reorder posterior columns to match alignment
        mapping = {r: _align_states(model, np.array([r]), dates[:1])[0]
                   for r in range(N_REGIMES)}
        inv_map = {v: k for k, v in mapping.items()}
        aligned_posteriors = np.zeros_like(posteriors)
        for sem_label in range(N_REGIMES):
            raw_label = inv_map[sem_label]
            aligned_posteriors[:, sem_label] = posteriors[:, raw_label]
        return aligned, aligned_posteriors
    else:
        return raw_states, posteriors


# ─── Metrics ──────────────────────────────────────────────────────────────────

def hmm_metrics(model: GaussianHMM, X: np.ndarray) -> dict:
    """
    Compute HMM-specific diagnostics.
    """
    ll = model.score(X)
    n_params = (
        N_REGIMES ** 2                  # transition matrix
        + N_REGIMES * X.shape[1]        # means
        + N_REGIMES * X.shape[1] ** 2   # full covariance
    )
    T = len(X)
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(T)

    return {
        "log_likelihood": round(ll, 4),
        "aic": round(aic, 2),
        "bic": round(bic, 2),
        "n_params": n_params,
        "converged": bool(model.monitor_.converged),
    }


def regime_transition_matrix(states: np.ndarray) -> pd.DataFrame:
    """Empirical month-over-month transition matrix from decoded states."""
    mat = np.zeros((N_REGIMES, N_REGIMES))
    for t in range(len(states) - 1):
        mat[states[t], states[t + 1]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / np.where(row_sums == 0, 1, row_sums)
    return pd.DataFrame(
        mat,
        index=[REGIME_NAMES[i] for i in range(N_REGIMES)],
        columns=[REGIME_NAMES[i] for i in range(N_REGIMES)],
    ).round(3)


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_hmm(model: GaussianHMM, path: str = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or os.path.join(MODEL_DIR, "hmm.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[HMM] Saved to {path}")


def load_hmm(path: str = None) -> GaussianHMM:
    path = path or os.path.join(MODEL_DIR, "hmm.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[HMM] Loaded from {path}")
    return model


if __name__ == "__main__":
    from data_fetcher import fetch_raw
    from features import build_features

    df = fetch_raw()
    X, dates, names, scaler = build_features(df)

    model = fit_hmm(X, dates)
    states, posteriors = hmm_predict(model, X, dates)
    metrics = hmm_metrics(model, X)

    print("\n--- HMM Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Empirical Transition Matrix ---")
    print(regime_transition_matrix(states))

    state_series = pd.Series(states, index=dates, name="regime")
    print("\n--- State counts ---")
    print(state_series.value_counts().sort_index().rename(REGIME_NAMES))
