import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from benchmarking.utils import split_data, naive_forecasts, calculate_metrics

from modularprophet.compositions import Additive
from modularprophet.components import Trend
from modularprophet.modules.seasonality import FourierSeasonality
from modularprophet.forecaster import ModularProphet


def evaluate_modularprophet(
    dataset, n_forecasts, n_lags, step_size, holdout, seasonal_cycle, config={}
):
    df = pd.read_csv(dataset)
    num_splits = (holdout - 2 * n_forecasts) + 1
    metrics = {}
    predictions = []

    for i in range(0, num_splits, step_size):
        train_df, eval_df, y = split_data(df, i, holdout, n_forecasts, n_lags)
        forecaster = ModularProphet(
            Additive(
                Trend(),
                FourierSeasonality(
                    "yearly", period=365.25, series_order=3
                ),  # multiply_with="Trend"
            )
        )
        _ = forecaster.fit(train_df, n_forecasts=n_forecasts, learning_rate=0.1)
        # Evaluation
        _, y, y_hat, _ = forecaster.predict(eval_df)
        prediction = y_hat.squeeze(0)
        predictions.append(
            np.pad(
                prediction,
                (i, num_splits - i),
                mode="constant",
                constant_values=(np.nan),
            )
        )
        # Metrics
        naive, seasonal_naive = naive_forecasts(train_df, seasonal_cycle, n_forecasts)
        metrics[i] = calculate_metrics(y, prediction, naive, seasonal_naive)
    return pd.DataFrame(metrics).T, pd.DataFrame(np.array(predictions)).T
