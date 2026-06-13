# XAUUSD Cross-Asset Model V2

An algorithmic trading system for **Gold (XAU/USD)** that predicts next-day directional drift using cross-asset market data, gated by a **3-state volatility-triggered Markov regime switch**. Executes trades via **MetaTrader 5**.

---

## Core Hypothesis

**Gold's daily return can be predicted one day ahead using a linear combination of:**

| Predictor | Source | Why it matters |
|---|---|---|
| Yesterday's Gold return | XAUUSD | Momentum/mean-reversion |
| Today's Dollar Index return | DXY | Inverse reserve-currency relationship |
| Today's S&P 500 return | SPX | Risk appetite / wealth effect |
| Today's change in US 10Y yield | US10Y | Opportunity cost of holding gold |
| Today's change in VIX | VIX | Fear / flight-to-safety |

**BUT** — the relationship between these predictors and gold's next-day return is **regime-dependent**. The market behaves fundamentally differently depending on whether volatility is low, medium, or high. The model first detects the volatility regime, then applies a regime-specific linear regression to generate the directional signal.

---

## Pipeline

### 1. Data Gathering (`1DATAGATHERING.py`)

Pulls daily OHLC data for 5 assets from **TradingView** (`tvDatafeed`):

| Asset | Symbol | Exchange | Data Used |
|---|---|---|---|
| Gold | GOLD | TVC | OHLC |
| US Dollar Index | DXY | TVC | Close |
| S&P 500 | SPX | SP | Close |
| US 10Y Treasury Yield | US10Y | TVC | Close |
| CBOE Volatility Index | VIX | CBOE | Close |

All 5 are inner-joined on date, keeping only days where all markets were open (~3,700 rows from 2010–present). Saved to `database/database.csv`.

### 2. Feature Engineering (`2FEATURE_ENGINEERING.ipynb`)

| Feature | Formula | Type |
|---|---|---|
| `XAU_Returns` | `ln(Close_t / Close_t-1)` | Log return (prices) |
| `DXY_Returns` | `ln(Close_t / Close_t-1)` | Log return (prices) |
| `SPX_Returns` | `ln(Close_t / Close_t-1)` | Log return (prices) |
| `US10Y_Diff` | `Close_t - Close_t-1` | Point difference (yields) |
| `VIX_Diff` | `Close_t - Close_t-1` | Point difference (VIX) |
| **Target** | `XAU_Returns.shift(-1)` | **Tomorrow's gold return** |

Prices use log returns (percentage-based). Yields and VIX use point differences because they can cross zero. The target is tomorrow's gold return — we predict forward.

Saved to `database/feature_matrix.csv`.

### 3. Regime Labelling (`3REGIME_LABELLING.ipynb`)

A **custom Markov regime model** classifies each day into one of 3 volatility states.

#### Calibration (first ~100 bars)
1. Compute daily volatility: `(High - Low) / Close`
2. Split into terciles at the 33rd and 67th percentiles
3. For each tercile, compute the mean and std of volatility → emission parameters for a Gaussian likelihood
4. Count transitions between regimes → apply additive smoothing (+0.1/+0.3) → transition matrix

#### Inference (every new bar) — Forward Algorithm
1. **Prior**: Multiply state probabilities by transition matrix: `prior = T' @ state_probs`
2. **Likelihood**: `P(vol | regime) = Gaussian(vol; mean_regime, std_regime)`
3. **Posterior**: `posterior = prior * likelihood / sum(...)`
4. **Argmax**: pick the most probable state

#### The 3 Regimes

| Regime | Label | Color | Meaning |
|---|---|---|---|
| 0 | **Low Volatility** | Green | Calm, orderly market |
| 1 | **Medium Volatility** | Amber | Normal conditions |
| 2 | **High Volatility** | Red | Panic / shock / spike |

The default transition matrix encodes **persistence** (90% probability of staying in the same state), with small probabilities for abrupt jumps.

Saved to `database/labeled_feature_matrix.csv`.

### 4. Data Partitioning (`4DATA_PARTITIONING.ipynb`)

Splits the labeled data into 3 regime-specific training files (features only):

| File | Regime | Rows |
|---|---|---|
| `0_LOW_VOL_TRAINING.csv` | Low Volatility | 1,515 |
| `1_MED_VOL_TRAINING.csv` | Medium Volatility | 1,240 |
| `2_HIGH_VOL_TRAINING.csv` | High Volatility | 950 |

### 5–7. Training (`5TRAINING0_LOW_VOL.ipynb`, `6TRAINING1_MID_VOL.ipynb`, `7TRAINING2_HIGH_VOL.ipynb`)

Each notebook trains `sklearn.linear_model.LinearRegression` on 70% of data (30% test split) — 5 features predicting `Target_Next_Day_Gold`.

#### Trained Coefficients

| Coefficient | Regime 0 (Low Vol) | Regime 1 (Mid Vol) | Regime 2 (High Vol) |
|---|---|---|---|
| **Intercept** | +0.000409 | +0.000026 | +0.000468 |
| **XAU_Returns** (lagged) | **+0.1305** | +0.0124 | +0.0152 |
| **DXY_Returns** | **+0.2075** | **+0.1218** | **-0.0484** |
| **SPX_Returns** | -0.0352 | +0.0273 | **-0.1145** |
| **US10Y_Diff** | +0.0069 | +0.0103 | -0.0009 |
| **VIX_Diff** | -0.0002 | +0.0001 | **-0.0007** |

**Key insight**: The signs and magnitudes flip across regimes:
- **DXY** is positively correlated with gold in Low/Mid vol but **negative** in High vol (classic risk-off: dollar and gold both safe havens)
- **SPX** is mildly negative in Low/High vol but positive in Mid vol
- In **High vol**, all coefficients except lagged Gold are negative — broad risk-off liquidation across all asset classes

Coefficients saved to `database/*_COEFFICIENTS.csv`.

---

## Live Trading Bot (`8APP.py`)

A production trading bot running on an hourly loop during the **London–New York overlap** (10:00–23:55 Nairobi time).

### Each Cycle
1. **Fetch live data**: Pull last 600 daily bars for all 5 assets from TradingView
2. **Compute features**: Log returns, diffs, and lagged gold return
3. **Calibrate Markov model**: On the last 100 bars of volatility data
4. **Detect regime**: Run forward inference on the most recent bar
5. **Generate signal**: Compute drift using regime-specific coefficients:

```
drift = intercept_regime
      + XAU_Returns_Lag1 * β_xau_regime
      + DXY_Returns      * β_dxy_regime
      + SPX_Returns      * β_spx_regime
      + US10Y_Diff       * β_us10y_regime
      + VIX_Diff         * β_vix_regime
```

6. **Trade rule**: If `|drift| > 0.0005` (5 bps threshold):
   - **BUY** if drift > +0.0005
   - **SELL** if drift < -0.0005

### Risk Management
| Parameter | Value |
|---|---|
| Risk per trade | 5% of account balance |
| Stop-loss | 1 × ATR(14) on H4 chart |
| Take-profit | 2 × ATR(14) on H4 chart |
| Max trades per day | 1 |
| Trading session | 10:00–23:55 Nairobi time (London–NY overlap) |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│               INPUT LAYER                    │
│  XAUUSD OHLC | DXY | SPX | US10Y | VIX     │
│             (TradingView API)                │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│           FEATURE ENGINEERING                │
│  XAU_Ret | DXY_Ret | SPX_Ret                │
│  US10Y_Δ  | VIX_Δ  | XAU_Ret_Lag1          │
└───────────────────┬─────────────────────────┘
          ┌────────┴────────┐
          ▼                 ▼
┌─────────────────┐  ┌─────────────────────────┐
│  REGIME DETECTOR│  │  REGIME-SPECIFIC MODELS │
│  (Markov Chain) │  │  (3× LinearRegression)  │
│                 │  │                         │
│  vol=(H-L)/C    │  │  Regime 0: Low Vol      │
│  calibrate(100) │  │  Regime 1: Mid Vol      │
│  forward_algo() │  │  Regime 2: High Vol     │
│  ↓ regime {0,1,2}│  │  5 features → drift    │
└────────┬────────┘  └───────────┬─────────────┘
         └──────────┬────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│            TRADE EXECUTION (MT5)            │
│  |drift| > 0.0005 → BUY / SELL              │
│  SL = 1×ATR(14), TP = 2×ATR(14)             │
│  5% risk, 1 trade/day, session-gated        │
└─────────────────────────────────────────────┘
```

---

## Key Design Choices & Risks

| Choice | Rationale | Risk |
|---|---|---|
| **Linear Regression** | Interpretable, stable, low overfit on small regime samples | Misses non-linear interactions |
| **Markov with Gaussian emissions** | Probabilistic, handles noise, gives confidence | Volatility distributions may not be Gaussian |
| **Daily timeframe** | Clean data, all markets aligned | Cannot react to intraday events |
| **One trade per day** | Disciplined, avoids overtrading | Misses multiple opportunities in volatile days |
| **Static coefficients in source** | No model file dependency | Requires manual retraining |
| **ATR(14) on H4 for stops** | Adaptive to recent volatility | H4 may not reflect same-day conditions |
| **5% fixed risk per trade** | Simple position sizing | Does not scale with volatility regime |

---

## Verification Strategy

To validate or improve this model:

1. **Walk-forward backtest**: Re-calibrate Markov model and retrain regressions on expanding/rolling windows instead of a static 2010–2025 split
2. **Benchmark**: Compare Sharpe ratio, max drawdown, win rate, and profit factor against buy-and-hold and a simple momentum strategy
3. **Regime stability**: Measure how often regimes flip; verify the Markov persistence assumption holds empirically
4. **Coefficient significance**: Check p-values of each coefficient — small coefficients (e.g., VIX_Diff in Low vol) may be noise
5. **Forward test**: Run the bot on a demo account before live deployment

---

## Files

| File | Purpose |
|---|---|
| `1DATAGATHERING.py` | Fetch daily data for all 5 assets from TradingView |
| `2FEATURE_ENGINEERING.ipynb` | Compute features and target |
| `3REGIME_LABELLING.ipynb` | Markov regime classification |
| `4DATA_PARTITIONING.ipynb` | Split data by regime |
| `5TRAINING0_LOW_VOL.ipynb` | Train Low Vol regime model |
| `6TRAINING1_MID_VOL.ipynb` | Train Mid Vol regime model |
| `7TRAINING2_HIGH_VOL.ipynb` | Train High Vol regime model |
| `8APP.py` | Live trading bot (MT5 execution) |
| `backtest.ipynb` | Backtesting (placeholder) |
| `database/` | All data and model coefficients |

---

*Developer: Jesse Njoroge Gitaka*
