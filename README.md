# 📈 Stock Price Estimator

A multimodal next-day **High**-price forecaster that fuses **historical market
data** with **tweet sentiment**. A per-stock Bi-LSTM+attention encoder turns the
price/indicator history into a feature vector, frozen BERT turns each day's
tweets into a sentiment vector, and a single pooled regressor fuses the two to
predict the next day's High — in **real dollars** — for any of the 25 stocks in
the dataset.

The dataset is [StockNet](https://github.com/yumoxu/stocknet-dataset): 25 tickers,
daily prices and tweets from **2014-01-02 to 2016-03-31**.

---

## 🧠 The three branches

| Branch | Question it answers | Model |
|--------|--------------------|-------|
| **Branch 1 — History** | What does the recent price/indicator trajectory imply? | Bi-LSTM + attention encoder → 128-d vector |
| **Branch 2 — Sentiment** | What were people saying the day before? | Frozen BERT [CLS] → daily-mean 768-d vector |
| **Branch 3 — Fusion** | Put them together and predict tomorrow's High | Pooled multimodal regressor → standardized High → $ |

Each branch is split into numbered steps (`branchXY` = branch `X`, step `Y`).

### Data flow

```
                         ┌──────────────────── PREPROCESSING (pp/) ───────────────────┐
 price/raw/*.csv ─┬─► extract_indicators ─► autoencoder(PCA) ─► pca_latent_indicators ─┐
                  │                                                                     │
                  └────────────────────────────────► mergeLatent ◄─────────────────────┘
                                                          │
                                                          ▼
                                          raw_pca_historical_csv/*.csv   ← single raw-$ source
                                                          │
 stocknet tweets ─► tweet_clean ─► processed_tweets/      │
                                                          │
                                                          ▼
   ┌──────── BRANCH 1 ────────┐        ┌──────── BRANCH 2 ────────┐
   │ branch11  train encoder  │        │ branch21  BERT embeddings│
   │ branch12  encode → 128-d │        │ branch22  daily ffill    │
   └────────────┬─────────────┘        └────────────┬─────────────┘
                │ pca_branch1_tensors                │ new_sentiment_embeddings
                └───────────────┬────────────────────┘
                                ▼
                ┌──────────────── BRANCH 3 ────────────────┐
                │ branch31  merge + scale → merged tensors  │
                │ branch32  train pooled regressor          │
                └─────────────────────┬─────────────────────┘
                                      ▼
                       pipeline/predict_stock.py
              (7-day forecast  OR  30-day held-out eval + plots)
```

---

## 📂 Directory layout

```
stock_estimator/
├── pp/                              # preprocessing
│   ├── tweet_clean.py               #   raw tweet JSON  → processed_tweets/
│   ├── extract_indicators.py        #   prices          → pp/indicators/
│   ├── autoencoder.py               #   indicators      → pca_latent_indicators/ (PCA, 5 dims)
│   ├── mergeLatent.py               #   prices+latents  → raw_pca_historical_csv/
│   └── indicators/                  #   per-stock technical-indicator CSVs
├── branches/
│   ├── _model.py                    # canonical model classes (shared by all branches)
│   ├── branch11.py / branch12.py    # Branch 1: train encoder / export 128-d features
│   ├── branch21.py / branch22.py    # Branch 2: BERT embeddings / daily forward-fill
│   └── branch31.py / 32.py           # Branch 3: merge / train regressor
├── pipeline/
│   ├── predict_stock.py             # 7-day forecast OR 30-day eval (inference entry point)
│   └── Results/plots/               # saved forecast PNGs  (Branch-3 *_model.pt here are legacy)
├── stock_data/
│   ├── stocknet-dataset-master/     # source tweets
│   ├── price/raw/                   # source prices (OHLCV)
│   ├── processed_tweets/            # cleaned tweets               (tweet_clean.py)
│   ├── pca_latent_indicators/       # 5 PCA indicator latents      (autoencoder.py)
│   ├── raw_pca_historical_csv/      # ★ raw-$ features, the source EVERY branch reads
│   ├── pca_historical_csv/          # legacy pre-normalized copy — NOT used (see invariant)
│   ├── new_sentiment_embeddings/    # daily 768-d BERT vectors     (branch21/22)
│   └── pca_branch1_tensors/         # 128-d encoder features       (branch12)
├── pca_new_merged_tensors/          # fused per-stock tensors      (branch31)
├── Results/pooled_model.pt          # trained pooled regressor     (branch32)
├── pca_best_model1.pth              # trained Branch-1 encoder + per-stock scalers (branch11)
└── final_trained_stock_model.pth    # encoder weights only (state_dict)
```

---

## 🔬 Pipeline, file by file

Every script has a multi-line module docstring with its exact inputs/outputs;
this is the high-level summary.

### Preprocessing (`pp/`)
1. **`tweet_clean.py`** — flattens the StockNet per-day tweet JSON and strips
   URLs / `@mentions` / `#hashtags` / `$` → `stock_data/processed_tweets/`.
2. **`extract_indicators.py`** — computes technical indicators (SMA, EMA, RSI,
   Stochastic, ADX, MACD, Bollinger Bands, OBV, rolling StdDev) with `pandas_ta`,
   oldest→newest so rolling windows only use the past → `pp/indicators/`.
3. **`autoencoder.py`** — *PCA* (despite the name) to 5 latent dims; scaler and
   PCA are fit on the training period only → `stock_data/pca_latent_indicators/`.
4. **`mergeLatent.py`** — adds `Movement_PerC = (Close−Open)/Open` and joins the
   5 latents onto the prices → **`stock_data/raw_pca_historical_csv/`**, the
   single raw-dollar feature source the rest of the pipeline consumes.

### Branch 1 — historical encoder
5. **`branch11.py`** — trains `SimplifiedHybridStockModel` to predict the next
   day's standardized High from a 30-day window. Saves the encoder **and the
   per-stock StandardScalers** (the single source of truth for un-scaling back to
   dollars) → `pca_best_model1.pth`.
6. **`branch12.py`** — runs `.encode()` over every 30-day window to export a
   128-d vector per day (test days excluded) → `stock_data/pca_branch1_tensors/`.

### Branch 2 — sentiment
7. **`branch21.py`** — frozen `bert-base-uncased`; mean of the daily [CLS]
   vectors → `stock_data/new_sentiment_embeddings/`.
8. **`branch22.py`** — reindexes onto a gap-free daily calendar and
   **forward-fills only** (no backfill = no leakage).

### Branch 3 — fusion, training, evaluation
9. **`branch31.py`** — joins each day's 128-d stock vector with the **prior
   day's** 768-d sentiment, scales (train-only fit) → `pca_new_merged_tensors/`.
10. **`branch32.py`** — pools all stocks, fits sentiment PCA (768→k, 95% train
    variance), trains `PooledMultimodalRegressor` with early stopping →
    `Results/pooled_model.pt`.

### Inference
11. **`pipeline/predict_stock.py`** — two modes:
    - *forecast* (default): next 7 trading days with per-stock bias correction.
    - *eval* (`--eval`): 30-day held-out test with no bias correction — the
      honest generalization score (replaces the old `branch33.py`).

### Models (`branches/_model.py`)
- **`SimplifiedHybridStockModel`** — Branch-1 encoder. `encode()` returns the
  128-d vector (mean+last-step attention pooling); `forward()` adds the scalar
  High head used only while training branch11.
- **`PooledMultimodalRegressor`** — Branch-3 head. Projects the stock and
  sentiment vectors, fuses them, and predicts standardized High; the caller
  un-scales with each stock's High mean/scale so one shared model recovers every
  stock's price level.

---

## ⚠️ The raw-data invariant (read before touching the pipeline)

**Every branch must read `stock_data/raw_pca_historical_csv` (real dollars).**

`branch11.py` fits its StandardScalers on the raw CSVs, so the scaler means are
dollar-scale (e.g. AAPL High ≈ \$107). There is a sibling
`stock_data/pca_historical_csv` that is a *pre-normalized* copy (High ≈ 0.06).
If a downstream branch reads the normalized copy and applies the raw scaler, it
feeds `(0.06 − 107) / 17.8 ≈ −6` garbage into the encoder and the output lands in
a meaningless unit instead of dollars. Keep all branches on the raw source; if
you change it, rebuild downstream in order **branch12 → branch31 → branch32**.

---

## 🚀 How to run

Use the global Python 3.12 (it has `torch 2.5.1+cu121`, CUDA-enabled) and run
**from the `stock_estimator/` directory** so relative paths resolve.

```bash
# 1) Preprocessing
python pp/tweet_clean.py
python pp/extract_indicators.py
python pp/autoencoder.py
python pp/mergeLatent.py

# 2) Branches (in order)
python branches/branch11.py     # train Branch-1 encoder
python branches/branch12.py     # export 128-d features
python branches/branch21.py     # BERT sentiment embeddings
python branches/branch22.py     # daily forward-fill
python branches/branch31.py     # merge features + sentiment
python branches/branch32.py     # train pooled regressor

# 3) Forecast any stock (next 7 trading days vs ground truth)
python pipeline/predict_stock.py            # defaults to AAPL
python pipeline/predict_stock.py MSFT JPM   # one or more tickers
python pipeline/predict_stock.py --all      # every stock + aggregate

# 4) Held-out 30-day evaluation (replaces old branch33)
python pipeline/predict_stock.py --eval     # all stocks, no bias correction
```

`predict_stock.py` prints a per-day table (predicted / actual / abs & % error +
RMSE, MAE, MAPE, directional accuracy, the persistence baseline) and saves a plot
to `pipeline/Results/plots/<STOCK>_7day_forecast.png`. With `--eval` it runs the
full 30-day held-out test and reports per-stock + aggregate metrics.

> **"Next 7 days"** = the final 7 trading days, which `branch11` held out of
> training — so they are genuinely unseen *and* we have the ground truth to plot.
> Each day is predicted one step ahead from the real 30-day window ending the day
> before; synthetic rows are never fed back in, so errors don't compound.

---

## 📊 Results

**Held-out test — final 30 unseen days, pooled across all 25 stocks** (`--eval`):

| RMSE | MAE | MAPE | Directional acc. |
|-----:|----:|-----:|-----------------:|
| \$2.65 | \$1.52 | 2.67% | 58.9% |

**7-day forecast, pooled across all 25 stocks** (`predict_stock.py --all`):

| | RMSE | MAPE |
|--|-----:|-----:|
| Raw model | \$3.22 | — |
| **+ bias correction** | **\$2.58** (−20%) | **1.54%** |

Most stocks forecast to within ~1% MAPE. Next-day High is dominated by a very
strong persistence baseline (yesterday's price), so the model only narrowly beats
or trails it on raw RMSE; its value-add is **directional signal** plus the
**bias correction**, which removes the pooled head's constant per-stock offset and
rescues the few stocks that were otherwise far off (e.g. HSBC RMSE \$3.78 → \$0.78).

---

## 🧱 Design choices that matter

- **Per-stock standardization + shared head.** Inputs are standardized per stock,
  so a single pooled regressor can't see absolute price levels; predictions are
  made in standardized space and un-scaled with each stock's own High mean/scale.
- **No leakage anywhere.** Scalers, PCA and indicators are all fit on the training
  period only; sentiment is forward-filled (never backfilled) and aligned to the
  **prior** day; the final 30 days stay unseen until `branch33`.
- **Bias correction (inference).** `predict_stock.py` estimates each stock's
  median residual over the 20 days *before* the forecast window and subtracts it —
  a leakage-free calibration that cuts 7-day RMSE ~20%.
- **GELU over ReLU** in the encoder/head so standardized negatives aren't zeroed.

---

## 🛠️ Dependencies

```bash
pip install torch pandas numpy scikit-learn tqdm pandas-ta matplotlib transformers
```
