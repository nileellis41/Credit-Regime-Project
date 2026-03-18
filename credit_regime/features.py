# features.py
"""
Feature engineering: momentum, level, z-scores, cross-series ratios.
Returns a clean (T, F) NumPy array alongside the aligned DatetimeIndex.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# ─── Helper ──────────────────────────────────────────────────────────────────

def _pct_change_safe(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods).replace([np.inf, -np.inf], np.nan)


def _rolling_zscore(s: pd.Series, window: int = 36) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / (sd + 1e-8)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame) -> Tuple[np.ndarray, pd.DatetimeIndex, list[str], StandardScaler]:
    """
    Build feature matrix from raw FRED DataFrame.

    Returns
    -------
    X           : np.ndarray  (T, n_features)  — scaled
    dates       : DatetimeIndex aligned to X rows
    feat_names  : list of feature names
    scaler      : fitted StandardScaler (for inverse transforms / inference)
    """
    df = df_raw.copy()

    feats = {}

    # ── Levels (z-scored over rolling 36m window for stationarity) ────────────
    feats["hy_oas_z"]       = _rolling_zscore(df["hy_oas"],    36)
    feats["ig_oas_z"]       = _rolling_zscore(df["ig_oas"],    36)
    feats["ci_tighten_z"]   = _rolling_zscore(df["ci_tighten"],36)
    feats["cc_delinq_z"]    = _rolling_zscore(df["cc_delinq"], 36)
    feats["mort_delinq_z"]  = _rolling_zscore(df["mort_delinq"],36)
    feats["unrate_z"]       = _rolling_zscore(df["unrate"],    36)
    feats["yield_curve_z"]  = _rolling_zscore(df["yield_curve"],36)

    # ── Momentum (MoM and 3M changes in spread levels) ────────────────────────
    feats["hy_oas_mom"]     = df["hy_oas"].diff(1)
    feats["hy_oas_3m"]      = df["hy_oas"].diff(3)
    feats["ig_oas_mom"]     = df["ig_oas"].diff(1)
    feats["yield_curve_mom"]= df["yield_curve"].diff(1)
    feats["unrate_mom"]     = df["unrate"].diff(1)

    # ── C&I Loan growth (YoY %) ───────────────────────────────────────────────
    feats["ci_loan_yoy"]    = _pct_change_safe(df["ci_loans"], 12) * 100

    # ── Cross-series: HY–IG basis (excess risk premium in credit) ─────────────
    feats["hy_ig_basis"]    = df["hy_oas"] - df["ig_oas"]
    feats["hy_ig_basis_z"]  = _rolling_zscore(feats["hy_ig_basis"], 36)

    # ── Composite stress score (equal-weight average of z-scores) ─────────────
    stress_components = ["hy_oas_z", "ci_tighten_z", "cc_delinq_z", "mort_delinq_z"]
    feats["stress_composite"] = pd.concat(
        [feats[c] for c in stress_components], axis=1
    ).mean(axis=1)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    feat_df = pd.DataFrame(feats, index=df.index)

    # Drop warm-up rows (36m rolling window + 12m YoY)
    feat_df = feat_df.dropna()

    feat_names = list(feat_df.columns)
    X_raw = feat_df.values.astype(np.float32)

    # Standard-scale (LSTM benefits from this; HMM works fine too)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    dates = feat_df.index

    print(f"[features] Built {len(feat_names)} features over {len(dates)} months "
          f"({dates[0].date()} → {dates[-1].date()})")

    return X_scaled, dates, feat_names, scaler


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of length `seq_len` across (X, y) to create
    overlapping sequences for the LSTM.

    Returns
    -------
    X_seq : (N, seq_len, n_features)
    y_seq : (N,)  — label of the LAST timestep in each window
    """
    Xs, ys = [], []
    for i in range(0, len(X) - seq_len, step):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


if __name__ == "__main__":
    from data_fetcher import fetch_raw
    df = fetch_raw()
    X, dates, names, scaler = build_features(df)
    print("Feature names:", names)
    print("X shape:", X.shape)
