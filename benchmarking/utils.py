import numpy as np
import pandas as pd


def calculate_metrics(y, y_hat, naive, seasonal_naive):
    mae = 1 / len(y) * np.sum(np.abs(y - y_hat))
    rmse = np.sqrt(1 / len(y) * np.sum((y - y_hat) ** 2))
    mase = 1 / len(y) * np.sum(np.abs(y - y_hat) / (1 / len(y) * np.sum(np.abs(y - naive))))
    smase = 1 / len(y) * np.sum(np.abs(y - y_hat) / (1 / len(y) * np.sum(np.abs(y - seasonal_naive))))
    mape = 1 / len(y) * np.sum(np.abs(y - y_hat) / y)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MASE": mase, "sMASE": smase}


def naive_forecasts(train_df, seasonal_cycle, n_forecasts):
    # Extract the last value from the training set and repeat it n_forecasts times
    naive = train_df["y"].tail(1).values * np.ones(n_forecasts)
    # Extract the last seasonal_cycle values from the training set and repeat them n_forecasts // seasonal_cycle + 1 times
    seasonal_naive = np.tile(train_df["y"].tail(seasonal_cycle).values, n_forecasts // seasonal_cycle + 1)[
        :n_forecasts
    ]
    return naive, seasonal_naive


def split_data(df, i, holdout, n_forecasts, n_lags):
    train_df = df[i : -holdout + n_forecasts + i]
    eval_df = df[-holdout + n_forecasts - n_lags + i :].head(n_forecasts + n_lags)
    y = eval_df["y"].values[-n_forecasts:]
    return train_df, eval_df, y


def store_dataframes(metrics, predictions, dataset, model_name):
    dataset = dataset.split("/")[-1].split(".")[0]
    metrics.to_csv(f"{dataset}/{model_name}_metrics.csv", index=False)
    predictions.to_csv(f"{dataset}/{model_name}_predictions.csv", index=False)


def calculate_component_attributions(components, y, naive, n_forecasts):
    cache = np.zeros(n_forecasts)
    baseline = naive
    for component in components.columns:
        cache += forecast[component].values
        mase = 1 - (1 / len(y) * np.sum(np.abs(y - cache) / (1 / len(y) * np.sum(np.abs(y - baseline)))))
        print(component, mase)
