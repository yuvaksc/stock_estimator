#!/usr/bin/env python
"""Forecast the next 7 trading days of a stock's High price and plot the
prediction against the ground truth -- for ANY stock in the dataset.

Run
---
    # from the stock_estimator/ directory (paths self-resolve, so any CWD works)
    python pipeline/predict_stock.py            # 7-day forecast, defaults to AAPL
    python pipeline/predict_stock.py MSFT JPM   # one or more tickers
    python pipeline/predict_stock.py --all      # 7-day forecast, every stock + aggregate
    python pipeline/predict_stock.py --eval     # 30-day held-out TEST eval, all stocks

Two modes
---------
* forecast (default): the next HORIZON (7) trading days with per-stock bias
  correction -- a forecasting view.
* evaluation (--eval): the model's raw one-step predictions over the full
  TEST_DAYS (30) held-out split with NO bias correction -- the honest score of
  how well the trained model generalizes (this replaces the old branch33.py).

What "next 7 days" means
------------------------
The model never saw the final 30 trading days of each stock (branch11 reserves
them as a held-out test split). We forecast the last 7 of those days, which are
the 7 days immediately after the model's known window -- i.e. its "next 7 days"
-- and because they are real historical days we also have the GROUND TRUTH to
plot against.

The full multimodal pipeline (all branches, raw-dollar data)
------------------------------------------------------------
    raw_pca_historical_csv  (real $ OHLCV + PCA latents)
        -> per-stock standardize (branch11 scaler)
        -> Branch-1 Bi-LSTM+Attention encoder .encode()  -> 128-d stock vector
    daily BERT tweet sentiment of the PRIOR day           -> 768-d
        -> branch31 merge-scaler -> branch32 sentiment PCA
    [stock 128-d | sentiment PCA]
        -> pooled multimodal regressor -> standardized High -> unscale to $

Each day is predicted ONE STEP AHEAD from the real 30-day window ending the day
before. We do not feed synthetic rows back in, so errors never compound across
the horizon (the previous version froze every feature except High and recursed,
which made the multi-day forecast degenerate).

Accuracy
--------
Two fixes do the heavy lifting:
  1. The whole pipeline now reads the SAME raw-dollar CSVs that branch11 fit its
     scalers on. Previously the downstream branches read a pre-normalized copy
     and applied the raw scaler to it, so the encoder saw ~(-6) garbage and the
     output was in a meaningless unit instead of dollars.
  2. A per-stock bias correction: the single pooled head leaves a roughly
     constant per-stock offset (the reason a few stocks had hugely negative R2).
     We estimate that offset as the median residual over the CALIB_DAYS trading
     days *before* the forecast window (no leakage) and subtract it. On the
     held-out 7-day window this cuts pooled RMSE by ~20%.
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Paths: resolve everything relative to the repo root (stock_estimator/) so the
# script runs from any working directory.
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "branches"))
from _metrics import format_metrics, regression_metrics  # noqa: E402
from _model import PooledMultimodalRegressor, SimplifiedHybridStockModel  # noqa: E402

DATA_DIR = os.path.join(ROOT, "stock_data", "raw_pca_historical_csv")
TWEET_DIR = os.path.join(ROOT, "stock_data", "new_sentiment_embeddings")
MERGED_DIR = os.path.join(ROOT, "pca_new_merged_tensors")
BRANCH1_CHECKPOINT = os.path.join(ROOT, "pca_best_model1.pth")
# branch32 writes here; fall back to the older pipeline/Results copy if needed.
POOLED_MODEL_PATH = os.path.join(ROOT, "Results", "pooled_model.pt")
if not os.path.exists(POOLED_MODEL_PATH):
    POOLED_MODEL_PATH = os.path.join(ROOT, "pipeline", "Results", "pooled_model.pt")
PLOTS_DIR = os.path.join(ROOT, "pipeline", "Results", "plots")

SEQ_LEN = 30        # input window length (must match branch11)
HORIZON = 7         # forecast this many trading days
CALIB_DAYS = 20     # residual window used to estimate the per-stock bias
TEST_DAYS = 30      # held-out evaluation horizon (matches branch11's reserved test split)
TARGET_COL = "High"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Show interactively on a TTY; otherwise render to PNG only.
SHOW_PLOTS = sys.stdout.isatty()
if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)


# ---------------------------------------------------------------------------
# Load the trained models once.
# ---------------------------------------------------------------------------
def load_models():
    branch1_ckpt = torch.load(BRANCH1_CHECKPOINT, map_location=device, weights_only=False)
    if "feature_scalers" not in branch1_ckpt:
        raise ValueError("branch11 checkpoint predates the leakage fix; rerun branch11.py")
    feature_cols = branch1_ckpt["feature_cols"]
    branch1 = SimplifiedHybridStockModel(input_dim=len(feature_cols)).to(device)
    branch1.load_state_dict(branch1_ckpt["state_dict"])
    branch1.eval()

    if not os.path.exists(POOLED_MODEL_PATH):
        raise FileNotFoundError("Missing pooled model; run branch32.py first")
    pooled_ckpt = torch.load(POOLED_MODEL_PATH, map_location=device, weights_only=False)
    regressor = PooledMultimodalRegressor(
        stock_dim=int(pooled_ckpt["stock_feature_dim"]),
        tweet_dim=int(pooled_ckpt["sentiment_dim"]),
        hidden=int(pooled_ckpt.get("hidden", 128)),
        dropout=float(pooled_ckpt.get("dropout", 0.3)),
    ).to(device)
    regressor.load_state_dict(pooled_ckpt["state"])
    regressor.eval()

    return {
        "branch1": branch1,
        "regressor": regressor,
        "feature_scalers": branch1_ckpt["feature_scalers"],
        "feature_cols": feature_cols,
        "high_idx": feature_cols.index(TARGET_COL),
        "pca_mean": np.asarray(pooled_ckpt["sentiment_pca_mean"], dtype=np.float32),
        "pca_components": np.asarray(pooled_ckpt["sentiment_pca_components"], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# One-step-ahead predictions for the last ``n_targets`` trading days.
# ---------------------------------------------------------------------------
def one_step_predictions(stock, n_targets, M):
    feature_cols = M["feature_cols"]
    scaler_state = M["feature_scalers"][stock]

    df = pd.read_csv(os.path.join(DATA_DIR, f"{stock}.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    raw_high = df[TARGET_COL].to_numpy()
    # Cast to float first so the int64 Volume column is not truncated on assign.
    scaled = df[feature_cols].astype("float64")
    scaled.loc[:, feature_cols] = (
        df[feature_cols].values - np.asarray(scaler_state["mean"])
    ) / np.asarray(scaler_state["scale"])

    required = SEQ_LEN + n_targets
    if len(df) < required:
        raise ValueError(f"{stock}: need {required} rows, found {len(df)}")
    feats = scaled.values[-required:]
    dates = df["Date"].to_numpy()[-required:]
    high_tail = raw_high[-required:]

    sequences, target_dates, actual, previous = [], [], [], []
    for i in range(n_targets):
        j = i + SEQ_LEN
        sequences.append(feats[i:j])
        target_dates.append(dates[j])
        actual.append(high_tail[j])
        previous.append(high_tail[j - 1])
    sequences = np.asarray(sequences, dtype=np.float32)
    target_dates = pd.to_datetime(target_dates)

    with torch.no_grad():
        stock_vec = M["branch1"].encode(
            torch.tensor(sequences, dtype=torch.float32, device=device)
        ).cpu().numpy()

    # Sentiment of the prior calendar day; forward-fill if a day is missing so
    # the forecast is robust for any stock/date.
    tweets = pd.read_csv(os.path.join(TWEET_DIR, f"sentiment_embeddings_{stock}.csv"))
    tweets["Date"] = pd.to_datetime(tweets["Date"])
    tweets = tweets.drop_duplicates("Date", keep="last").set_index("Date")
    sentiment_dates = target_dates - pd.Timedelta(days=1)
    aligned = tweets.reindex(sentiment_dates)
    if aligned.isna().any().any():
        aligned = (
            tweets.reindex(tweets.index.union(sentiment_dates))
            .sort_index().ffill().reindex(sentiment_dates)
        )
    tweet_vec = aligned.to_numpy(dtype=np.float32)

    # Standardize the merged [stock | sentiment] vector with branch31's scaler.
    meta = torch.load(os.path.join(MERGED_DIR, f"{stock}.pt"), map_location="cpu", weights_only=False)
    combined = np.hstack([stock_vec, tweet_vec])
    combined = (combined - np.asarray(meta["scaler_mean"])) / np.asarray(meta["scaler_scale"])
    stock_scaled = combined[:, :128].astype(np.float32)
    tweet_scaled = combined[:, 128:].astype(np.float32)
    tweet_pca = (tweet_scaled - M["pca_mean"]) @ M["pca_components"].T

    with torch.no_grad():
        pred_std = M["regressor"](
            torch.tensor(stock_scaled, dtype=torch.float32, device=device),
            torch.tensor(tweet_pca, dtype=torch.float32, device=device),
        ).cpu().numpy()

    hm = float(scaler_state["mean"][M["high_idx"]])
    hs = float(scaler_state["scale"][M["high_idx"]])
    predicted = pred_std * hs + hm  # unscale to raw dollars

    return target_dates, np.asarray(actual), np.asarray(predicted), np.asarray(previous)


# ---------------------------------------------------------------------------
# Build the 7-day forecast for one stock (with bias correction + metrics).
# ---------------------------------------------------------------------------
def forecast_stock(stock, M):
    dates, actual, predicted, previous = one_step_predictions(stock, CALIB_DAYS + HORIZON, M)

    # Per-stock bias from the calibration window that precedes the forecast.
    cal_pred, cal_act = predicted[:CALIB_DAYS], actual[:CALIB_DAYS]
    bias = float(np.median(cal_pred - cal_act))

    f_dates = dates[CALIB_DAYS:]
    f_actual = actual[CALIB_DAYS:].astype(np.float64)
    # float64 so the table rounds cleanly (model output is float32).
    f_pred = (predicted[CALIB_DAYS:] - bias).astype(np.float64)
    f_prev = previous[CALIB_DAYS:].astype(np.float64)

    # Reuse the shared metric set; add the two forecast-only extras.
    metrics = regression_metrics(f_actual, f_pred, f_prev)
    metrics["persistence_rmse"] = float(np.sqrt(np.mean((f_actual - f_prev) ** 2)))
    metrics["bias"] = bias

    table = pd.DataFrame({
        "date": f_dates.date,
        "predicted": np.round(f_pred, 2),
        "actual": np.round(f_actual, 2),
        "abs_err": np.round(np.abs(f_actual - f_pred), 2),
        "pct_err": np.round(np.abs((f_actual - f_pred) / f_actual) * 100, 2),
    })
    context = (dates[:CALIB_DAYS], actual[:CALIB_DAYS])  # recent history for the plot
    return table, metrics, context, (f_dates, f_actual, f_pred)


def plot_forecast(stock, metrics, context, forecast):
    ctx_dates, ctx_actual = context
    f_dates, f_actual, f_pred = forecast

    plt.figure(figsize=(9, 4.5))
    plt.plot(ctx_dates, ctx_actual, color="0.6", lw=1.2, label="Recent actual")
    plt.plot(f_dates, f_actual, "o-", color="#2ca02c", lw=2, label="Actual (ground truth)")
    plt.plot(f_dates, f_pred, "s--", color="#ff7f0e", lw=2, label="Predicted")
    plt.axvline(f_dates[0], color="0.8", ls=":", lw=1)
    plt.title(f"{stock}: next {HORIZON} trading days  "
              f"(RMSE ${metrics['rmse']:.2f}, MAPE {metrics['mape']:.2f}%, "
              f"dir {metrics['directional']:.0f}%)")
    plt.xlabel("Date")
    plt.ylabel("High price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()  # rotate date labels so they don't overlap
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"{stock}_7day_forecast.png")
    plt.savefig(out, dpi=120)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Held-out TEST evaluation: the model's RAW one-step predictions over the full
# TEST_DAYS split, with no bias correction -- the honest generalization score.
# (This is what branch33.py used to do, now folded into the pipeline.)
# ---------------------------------------------------------------------------
def evaluate_stock(stock, M):
    dates, actual, predicted, previous = one_step_predictions(stock, TEST_DAYS, M)
    actual = actual.astype(np.float64)
    predicted = predicted.astype(np.float64)
    previous = previous.astype(np.float64)
    metrics = regression_metrics(actual, predicted, previous)
    return dates, actual, predicted, previous, metrics


def plot_eval(stock, dates, actual, predicted, metrics):
    plt.figure(figsize=(9, 4.5))
    plt.plot(dates, actual, "o-", color="#2ca02c", lw=1.8, label="Actual (ground truth)")
    plt.plot(dates, predicted, "s--", color="#ff7f0e", lw=1.8, label="Predicted")
    plt.title(f"{stock}: {TEST_DAYS}-day held-out test  "
              f"(RMSE ${metrics['rmse']:.2f}, MAPE {metrics['mape']:.2f}%, "
              f"dir {metrics['directional']:.0f}%)")
    plt.xlabel("Date")
    plt.ylabel("High price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"{stock}_test.png")
    plt.savefig(out, dpi=120)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    return out


def has_artifacts(stock, valid):
    """True if the stock has the trained scaler, raw CSV and merged tensor."""
    if stock not in valid or not os.path.exists(os.path.join(DATA_DIR, f"{stock}.csv")):
        print(f"Skipping {stock}: no trained scaler / data")
        return False
    if not os.path.exists(os.path.join(MERGED_DIR, f"{stock}.pt")):
        print(f"Skipping {stock}: missing merged tensor (run branches 12/31)")
        return False
    return True


def available_stocks():
    return sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))


def main():
    args = sys.argv[1:]
    eval_mode = "--eval" in args
    run_all = "--all" in args
    flags = {"--eval", "--all"}
    tickers = [a.upper() for a in args if a not in flags]

    M = load_models()
    valid = set(M["feature_scalers"])

    if eval_mode:
        _run_eval(M, valid)
    else:
        stocks = available_stocks() if run_all else (tickers or ["AAPL"])
        _run_forecast(stocks, M, valid, run_all)


def _run_eval(M, valid):
    all_actual, all_pred, all_prev = [], [], []
    for stock in available_stocks():
        if not has_artifacts(stock, valid):
            continue
        dates, actual, predicted, previous, m = evaluate_stock(stock, M)
        plot_eval(stock, dates, actual, predicted, m)
        print(f"{stock}: held-out days={len(actual)}, "
              f"{dates.min().date()} to {dates.max().date()} | "
              f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
              f"MAPE={m['mape']:.2f}%  Dir={m['directional']:.2f}%")
        all_actual.append(actual)
        all_pred.append(predicted)
        all_prev.append(previous)

    if all_actual:
        A = np.concatenate(all_actual)
        P = np.concatenate(all_pred)
        PR = np.concatenate(all_prev)
        m = regression_metrics(A, P, PR)
        print(f"\n=== AGGREGATE held-out TEST (pooled across {len(all_actual)} stocks, "
              f"{len(A)} samples) ===")
        print(format_metrics(m))
        print(f"  plots saved to {PLOTS_DIR}")


def _run_forecast(stocks, M, valid, run_all):
    agg_actual, agg_pred, agg_prev = [], [], []
    for stock in stocks:
        if not has_artifacts(stock, valid):
            continue

        table, metrics, context, forecast = forecast_stock(stock, M)
        out = plot_forecast(stock, metrics, context, forecast)

        if run_all:
            print(f"{stock:5s} RMSE ${metrics['rmse']:6.3f} "
                  f"(persistence ${metrics['persistence_rmse']:6.3f}) "
                  f"MAPE {metrics['mape']:5.2f}%  dir {metrics['directional']:5.1f}%  "
                  f"bias ${metrics['bias']:+.2f}")
        else:
            print(f"\n========== {stock}: next {HORIZON} trading days ==========")
            print(table.to_string(index=False))
            print(f"\n  RMSE              : ${metrics['rmse']:.3f}")
            print(f"  MAE               : ${metrics['mae']:.3f}")
            print(f"  MAPE              : {metrics['mape']:.2f}%")
            print(f"  Directional acc.  : {metrics['directional']:.1f}%")
            print(f"  Persistence RMSE  : ${metrics['persistence_rmse']:.3f} (baseline)")
            print(f"  Bias correction   : ${metrics['bias']:+.3f}")
            print(f"  Plot saved to     : {out}")

        _, f_actual, f_pred = forecast
        agg_actual.append(f_actual)
        agg_pred.append(f_pred)
        agg_prev.append(metrics)

    if run_all and agg_actual:
        A = np.concatenate(agg_actual)
        P = np.concatenate(agg_pred)
        rmse = np.sqrt(np.mean((A - P) ** 2))
        mape = np.mean(np.abs((A - P) / A)) * 100
        print(f"\n=== AGGREGATE 7-day forecast across "
              f"{len(agg_actual)} stocks ({len(A)} samples) ===")
        print(f"  RMSE ${rmse:.4f}   MAPE {mape:.2f}%")
        print(f"  plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
