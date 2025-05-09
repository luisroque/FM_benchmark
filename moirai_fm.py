import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

TIME_COL = 'ds'

DATASET_GROUP_FREQ = {
    "Tourism": {
        "Monthly": {"FREQ": "M", "H": 24},
    },
    "M1": {
        "Monthly": {"FREQ": "M", "H": 24},
        "Quarterly": {"FREQ": "Q", "H": 8},
    },
    "M3": {
        "Monthly": {"FREQ": "M", "H": 24},
        "Quarterly": {"FREQ": "Q", "H": 8},
        "Yearly": {"FREQ": "Y", "H": 4},
    },
    "M4": {
        "Monthly": {"FREQ": "M", "H": 24},
        "Quarterly": {"FREQ": "Q", "H": 8},
    },
    "Traffic": {
        "Daily": {"FREQ": "D", "H": 30},
    },
    "M5": {
        "Daily": {"FREQ": "D", "H": 60},
    },
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

results = []

for model_size in ("small", "large"):
    print(f"\n\n\n\nStarting for model size: {model_size}\n\n\n\n")
    for f in features_files:
        print(f"\nProcessing: {f}")

        # Extract dataset group and frequency type from the filename
        filename = os.path.basename(f).replace("_features.pkl", "")
        group_name, freq_type = filename.split("_")

        # Get freq and horizon
        config = DATASET_GROUP_FREQ[group_name][freq_type]
        FREQ = config["FREQ"]
        FORECAST_HORIZON = config["H"]

        # Load data
        dataset = joblib.load(f)
        df = dataset["test_long"]
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        print(f"Distinct number of time series: {len(df['unique_id'].unique())}")

        # Create GluonTS dataset
        ds = PandasDataset.from_long_dataframe(
            df.set_index(TIME_COL),
            item_id="unique_id",
            target="y",
            freq=FREQ
        )

        # Split into train and test
        train, test_template = split(ds, offset=-FORECAST_HORIZON)
        test_data = test_template.generate_instances(
            prediction_length=FORECAST_HORIZON,
            windows=1,
            distance=FORECAST_HORIZON
        )

        df = df.sort_values(["unique_id", TIME_COL])

        train_df = df.groupby("unique_id", group_keys=False).apply(lambda g: g.iloc[:-FORECAST_HORIZON])
        test_df = df.groupby("unique_id", group_keys=False).apply(lambda g: g.iloc[-FORECAST_HORIZON:])

        print("Train/Test shapes:", train_df.shape, test_df.shape)

        # Load Moirai model
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{model_size}"),
            prediction_length=FORECAST_HORIZON,
            context_length=3000,
            patch_size='auto',
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
        predictor = model.create_predictor(batch_size=32)
        forecasts = predictor.predict(test_data.input)

        # Generate forecast dataframe
        test_pred = []
        for forecast in forecasts:
            mean_forecast = forecast.samples.mean(axis=0)
            dates = pd.date_range(start=forecast.start_date.to_timestamp(), periods=len(mean_forecast), freq=FREQ)
            df_temp = pd.DataFrame({
                'unique_id': forecast.item_id,
                'ds': dates,
                'y': mean_forecast
            })
            test_pred.append(df_temp)

        test_df_pred = pd.concat(test_pred).reset_index(drop=True)
        df.rename(columns={'y': 'y_true'}, inplace=True)

        final_df = df.merge(test_df_pred, on=["unique_id", "ds"], how="left")


        # MASE function
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


        SEASONALITY_MAP = {"M": 12, "Q": 4, "Y": 1, "D": 365}
        mase_seasonality = SEASONALITY_MAP[FREQ]

        mase_series = final_df.groupby("unique_id").apply(
            lambda df_mase: mase(
                df_mase["y_true"],
                df_mase["y"],
                m=mase_seasonality,
                h=FORECAST_HORIZON,
            )
        )

        mean_mase = mase_series.mean()
        print(f"Mean MASE: {mean_mase:.4f}")

        results.append({
            "model_size": model_size,
            "dataset": group_name,
            "frequency": freq_type,
            "FREQ": FREQ,
            "H": FORECAST_HORIZON,
            "num_series": len(df["unique_id"].unique()),
            "mean_mase": mean_mase
        })

        results_df = pd.DataFrame(results)
        results_df.to_csv("moirai_mase_results.csv", index=False)
        print("\nSaved MASE results to 'moirai_mase_results.csv'")

        # Plot 8 sample series
        # df_filtered = final_df.copy()
        # df_filtered = df_filtered.dropna(subset=["y"])
        # selected_ids = df_filtered['unique_id'].unique()[:4]
        #
        # plt.figure(figsize=(16, 12))
        # for i, uid in enumerate(selected_ids, 1):
        #     subset = df_filtered[df_filtered['unique_id'] == uid]
        #     plt.subplot(2, 2, i)
        #     plt.plot(subset['ds'], subset['y_true'], label='y_true')
        #     plt.plot(subset['ds'], subset['y'], label='y (forecast)')
        #     plt.title(f'Series: {uid}')
        #     plt.xlabel('Date')
        #     plt.ylabel('Value')
        #     plt.legend()
        #
        # plt.tight_layout()
        # plt.show()
