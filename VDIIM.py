import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

def load_and_process_data():
    df = pd.read_csv("Nifty50dataset.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    df = df[["open", "high", "low", "close"]]
    df = df.between_time("09:15", "15:30")

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["rolling_vol_15"] = df["log_return"].rolling(window=15).std()
    df["rolling_vol_15"] = df["rolling_vol_15"].shift(1)

    percentile_threshold = df["rolling_vol_15"].quantile(0.95)
    df["vol_spike"] = df["rolling_vol_15"] > percentile_threshold

    df["range"] = df["high"] - df["low"]
    df["range"] = df["range"].shift(1)
    
    df.dropna(inplace=True)

    # DAILY SUMMARY
    daily = df.resample("D").agg({
        "rolling_vol_15": ["mean", "max"],
        "vol_spike": "sum"
    })

    daily.columns = ["avg_vol", "max_vol", "spike_count"]
    daily = daily.dropna()

    # REGIME CLASSIFICATION
    low_thresh = daily["spike_count"].quantile(0.2)
    high_thresh = daily["spike_count"].quantile(0.8)

    def classify(x):
        if x <= low_thresh:
            return "Low Vol"
        elif x >= high_thresh:
            return "High Vol"
        else:
            return "Normal"

    daily["regime"] = daily["spike_count"].apply(classify)

    # Adding Hypothesis Test
    df["date"] = df.index.date
    daily["date"] = daily.index.date

    merged = df.merge(daily[["date", "regime"]], on = "date")

    high_vol = merged[merged["regime"] == "High Vol"]["range"]
    normal_vol = merged[merged["regime"] == "Normal"]["range"]

    t_stat, p_value = ttest_ind(high_vol, normal_vol, equal_var=False)

    hypothesis_result = {
        "t_stat" : float(t_stat),
        "p_value" : float(p_value),
        "high_vol_mean_range" : float(high_vol.mean()),
        "normal_vol_mean_range" : float(normal_vol.mean()),
    }

    df["open_lag1"] = df["open"].shift(1)
    df["close_lag1"] = df["close"].shift(1)
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_15"] = df["close"].rolling(15).mean()
    df["ma_diff"] = df["ma_5"] - df["ma_15"]

    # EMA FEATURES
    df["ema_8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema_30"] = df["close"].ewm(span=30, adjust=False).mean()

    # OPTIONAL: crossover signal (very useful)
    df["ema_diff"] = df["ema_8"] - df["ema_30"]

    df["ema_8"] = df["ema_8"].shift(1)
    df["ema_30"] = df["ema_30"].shift(1)
    df["ema_diff"] = df["ema_diff"].shift(1)

    df["next_open"] = df["open"].shift(-1)

    df_model = df.dropna()

    features = [
    "open", "high", "low", "close",
    "open_lag1", "close_lag1",
    "rolling_vol_15",
    "range",
    "return_1", "return_3",
    "ma_diff", 
    "ema_8", "ema_30", "ema_diff"
]

    X = df_model[features]
    y = df_model["next_open"]

    tscv = TimeSeriesSplit(n_splits=5)
    model = LinearRegression()

    mae_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mae_scores.append(mae)

    avg_mae = np.mean(mae_scores)

    df.dropna(inplace=True)
    
    
    features += ["open_lag1", "close_lag1"]

    # Train on full data after validation
    model.fit(X, y)

    latest_features = X.iloc[-1:].values
    next_open_pred = float(model.predict(latest_features)[0])

    regression_result = {
        "avg_mae": float(avg_mae),
        "predicted_next_open": next_open_pred
    }

    return df, daily, hypothesis_result, regression_result