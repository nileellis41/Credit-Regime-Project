# Credit Regime Detector — LSTM vs HMM

Detects 3-state credit market regimes (Benign / Stress / Crisis) from FRED macro series.
Compares a GaussianHMM baseline against a 2-layer LSTM classifier.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your FRED API key
cp .env.example .env
# edit .env → FRED_API_KEY=your_key_here
# Free key: https://fred.stlouisfed.org/docs/api/api_key.html

# 3. Run the full pipeline
python main.py

# Force-refresh FRED data (bypasses cache)
python main.py --no-cache

# Load already-trained models
python main.py --load
```

---

## Architecture

```
FRED API  →  data_fetcher.py  →  features.py  →  X: (T, 18 features)
                                                       │
                                               ┌───────┴───────┐
                                           hmm_model.py    lstm_model.py
                                         GaussianHMM(3)   LSTM(2L, 128h)
                                           Viterbi →          ↑
                                           pseudo-labels  supervised on
                                                          HMM labels
                                               └───────┬───────┘
                                                  evaluation.py
                                              agreement / timing / entropy
```

### FRED Series Used

| Feature Group     | Series ID         | Description                          |
|-------------------|-------------------|--------------------------------------|
| Credit spreads    | BAMLH0A0HYM2      | ICE BofA HY Option-Adjusted Spread   |
|                   | BAMLC0A0CM        | ICE BofA IG Corporate OAS            |
| Lending standards | DRTSCILM          | % banks tightening C&I standards     |
|                   | CILACBW027SBOG    | C&I loans outstanding                |
| Delinquency       | DRCCLACBS         | Credit card delinquency rate         |
|                   | DRSFRMACBS        | SF mortgage delinquency rate         |
| Macro             | UNRATE            | Unemployment rate                    |
|                   | T10Y2Y            | 10Y–2Y treasury spread               |

### Engineered Features (18 total)
- Rolling 36m z-scores of all levels
- 1M and 3M momentum (OAS, yield curve, unemployment)
- C&I loan YoY growth
- HY–IG basis and its z-score
- Composite stress score (equal-weight average of stress z-scores)

---

## LSTM Architecture

```
Input: (batch, 24 months, 18 features)
  → LSTM(hidden=128, layers=2, dropout=0.3)
  → LayerNorm(128)
  → Dropout(0.3)
  → Linear(128 → 3)
  → Softmax
```

- **Loss**: class-weighted cross-entropy (label smoothing=0.05)
- **Optimizer**: AdamW + CosineAnnealingLR
- **Split**: chronological train/val/test (70/15/15)
- **Early stopping**: patience=20 on val loss

---

## Outputs (`results/`)

| File                     | Contents                                         |
|--------------------------|--------------------------------------------------|
| `regime_comparison.png`  | 4-panel: OAS + regime shading, posteriors, agreement |
| `confusion_matrices.png` | LSTM test confusion + HMM vs LSTM agreement matrix |
| `regime_stats.csv`       | Per-regime macro statistics (both models)        |
| `metrics_summary.txt`    | Full text comparison report                      |

---

## Key Design Decisions

**Why HMM labels to train LSTM?**  
The LSTM and HMM share no labels from the environment — there's no ground truth "regime" time series. Using the HMM's Viterbi path as pseudo-labels lets us train LSTM as a classifier, enabling a fair architectural comparison on the same label space. The STRESS_ANCHORS in `config.py` semantically pin both models' state assignments to known historical regimes.

**Chronological split**  
Financial time series must not be shuffled. Train uses 2000–~2018, val ~2018–2020, test ~2020–present (dates depend on your pull).

**Rolling z-scores over raw levels**  
Spreads and delinquencies are non-stationary. Rolling z-scores (36m window) give the HMM Gaussian-emission assumptions a fighting chance while preserving regime-relative information for the LSTM.
