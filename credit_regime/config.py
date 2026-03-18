# config.py
"""
Central configuration for the LSTM Credit Regime Detector.
Set your FRED API key in a .env file: FRED_API_KEY=your_key_here
Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── FRED ──────────────────────────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # fallback to empty; will raise clearly

FRED_SERIES = {
    # Credit spreads
    "hy_oas":       "BAMLH0A0HYM2",       # ICE BofA HY Option-Adjusted Spread (bp)
    "ig_oas":       "BAMLC0A0CM",          # ICE BofA IG Corporate OAS (bp)
    # Lending conditions
    "ci_tighten":   "DRTSCILM",            # % banks tightening C&I standards (quarterly → ffill)
    "ci_loans":     "CILACBW027SBOG",      # C&I loans outstanding, all commercial banks ($B)
    # Delinquency rates
    "cc_delinq":    "DRCCLACBS",           # Credit card delinquency rate (%)
    "mort_delinq":  "DRSFRMACBS",          # SF mortgage delinquency rate (%)
    # Macro context
    "unrate":       "UNRATE",              # Civilian unemployment rate (%)
    "yield_curve":  "T10Y2Y",              # 10Y-2Y treasury spread (%)
}

START_DATE  = "2000-01-01"
END_DATE    = None          # None = pull through today
RESAMPLE_FREQ = "MS"        # month-start

# ── Regime labels ─────────────────────────────────────────────────────────────
N_REGIMES = 3
REGIME_NAMES = {0: "Benign", 1: "Stress", 2: "Crisis"}
REGIME_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

# Known stress anchors used to semantically align HMM states
# (month, year, expected regime)
STRESS_ANCHORS = [
    ("2008-10", 2),   # GFC peak
    ("2020-04", 2),   # COVID shock
    ("2022-06", 1),   # rate shock / tightening
    ("2004-06", 0),   # benign mid-cycle
    ("2017-01", 0),   # benign expansion
]

# ── LSTM architecture ─────────────────────────────────────────────────────────
SEQUENCE_LEN    = 24        # months of look-back
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.3
BIDIRECTIONAL   = False

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
MAX_EPOCHS      = 150
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
PATIENCE        = 20        # early stopping patience
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_ITER      = 200
HMM_COV_TYPE    = "full"    # full | diag | tied | spherical

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
MODEL_DIR       = "models"
RESULTS_DIR     = "results"
