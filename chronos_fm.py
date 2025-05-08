import os
import joblib
import torch
import pandas as pd
import numpy as np
from chronos import ChronosPipeline

TIME_COL = "ds"
TARGET = "y"
SEASONALITY_MAP = {"M": 12, "Q": 4, "Y": 1, "D": 7}

DATASET_GROUP_FREQ = {
    "Tourism": {"Monthly": {"FREQ": "M", "H": 24}},
    "M1": {"Monthly": {"FREQ": "M", "H": 24}, "Quarterly": {"FREQ": "Q", "H": 8}},
    "M3": {"Monthly": {"FREQ": "M", "H": 24}, "Quarterly": {"FREQ": "Q", "H": 8}, "Yearly": {"FREQ": "Y", "H": 4}},
    "M4": {"Monthly": {"FREQ": "M", "H": 24}, "Quarterly": {"FREQ": "Q", "H": 8}},
    "Traffic": {"Daily": {"FREQ": "D", "H": 30}},
    "M5": {"Daily": {"FREQ": "D", "H": 60}},
}

features_files = [
    "processed_datasets/M1_Monthly_features.pkl",
    "processed_datasets/M1_Quarterly_features.pkl",
    "processed_datasets/M3_Monthly_features.pkl",
    "processed_datasets/M3_Quarterly_features.pkl",
    "processed_datasets/M3_Yearly_features.pkl",
    "processed_datasets/M4_Monthly_features.pkl",
    "processed_datasets/M4_Quarterly_features.pkl",
    "processed_datasets/M5_Daily_features.pkl",
    "processed_datasets/Tourism_Monthly_features.pkl",
    "processed_datasets/Traffic_Daily_features.pkl"
]

pipeline_tiny = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny", device_map="cpu", torch_dtype=torch.bfloat16)
pipeline_large = ChronosPipeline.from_pretrained("amazon/chronos-t5-large", device_map="cpu", torch_dtype=torch.bfloat16)

def mase(y_true, y_pred, h, m: int = 1):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.size <= max(m, h):
        return np.nan
    y_true_insample = y_true[:-h]
    y_true_h = y_true[-h:]
    y_pred_h = y_pred[-h:]
    mask = ~np.isnan(y_pred_h)
    if mask.sum() == 0:
        return np.nan
    scale = np.mean(np.abs(y_true_insample[m:] - y_true_insample[:-m]))
    if scale == 0.0 or np.isnan(scale):
        scale = np.mean(np.abs(y_true_insample[1:] - y_true_insample[:-1]))
        if scale == 0.0 or np.isnan(scale):
            return np.nan
    mase_value = np.mean(np.abs(y_true_h - y_pred_h)) / scale
    return float(mase_value)


def chronos_forecast(model, data, horizon, target):
    context = torch.tensor(data[-horizon:][target].tolist())
    forecast = model.predict(context, horizon)
    return np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)


def convert_forecast_to_pandas(forecast, holdout_set):
    forecast_pd = holdout_set[["unique_id", "ds"]].copy()
    forecast_pd["forecast_lower"] = forecast[0]
    forecast_pd["forecast"] = forecast[1]
    forecast_pd["forecast_upper"] = forecast[2]
    return forecast_pd


results = []

for f in features_files:
    print(f"\nProcessing: {f}")
    filename = os.path.basename(f).replace("_features.pkl", "")
    group_name, freq_type = filename.split("_")
    config = DATASET_GROUP_FREQ[group_name][freq_type]
    FREQ = config["FREQ"]
    H = config["H"]
    mase_seasonality = SEASONALITY_MAP[FREQ]

    # load data
    dataset = joblib.load(f)
    df = dataset["test_long"]
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(["unique_id", TIME_COL])
    all_ids = df['unique_id'].unique()

    # train/test split
    train_df = df.groupby("unique_id", group_keys=False).apply(lambda g: g.iloc[:-H])
    test_df = df.groupby("unique_id", group_keys=False).apply(lambda g: g.iloc[-H:])
    print("Train/Test shapes:", train_df.shape, test_df.shape)

    for model_size, pipeline in [("tiny", pipeline_tiny), ("large", pipeline_large)]:
        print(f"Running Chronos-{model_size} on {filename}")
        forecasts = []

        for i, ts in enumerate(all_ids, 1):
            print(f"[{i}/{len(all_ids)}] Forecasting for series: {ts}")
            test_data = test_df[test_df['unique_id'] == ts]
            train_data = train_df[train_df['unique_id'] == ts]
            lower, mid, upper = chronos_forecast(pipeline, train_data, H, TARGET)
            forecast_df = convert_forecast_to_pandas([lower, mid, upper], test_data)
            forecasts.append(forecast_df)

        forecast_df = pd.concat(forecasts).reset_index(drop=True)

        # merge with actuals
        merged = df.merge(forecast_df[["unique_id", "ds", "forecast"]], on=["unique_id", "ds"], how='left')
        merged.sort_values(by=["unique_id", "ds"], inplace=True)
        mase_series = merged.groupby("unique_id").apply(lambda g: mase(g["y"], g["forecast"], h=H, m=mase_seasonality))
        mean_mase = mase_series.mean()
        print(f"Mean MASE ({model_size}): {mean_mase:.4f}")

        results.append({
            "model_size": model_size,
            "dataset": group_name,
            "frequency": freq_type,
            "FREQ": FREQ,
            "H": H,
            "num_series": len(all_ids),
            "mean_mase": mean_mase
        })

results_df = pd.DataFrame(results)
results_df.to_csv("chronos_mase_results.csv", index=False)
print("\nSaved Chronos MASE results to 'chronos_mase_results.csv'")
