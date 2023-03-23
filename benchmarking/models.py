import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from benchmarking.utils import split_data, naive_forecasts, calculate_metrics

from modularprophet.forecaster import ModularProphet


def calculate_component_attributions(components, y, n_lags):
    components = {k: v for k, v in components.items() if k != "y_hat"}
    r2 = {}
    y = y[n_lags:]
    cache = np.zeros_like(y, dtype=np.float32)
    for name, component in components.items():
        # extract 1-step ahead forecasts
        cache += np.concatenate(
            [
                np.array([c[0] for c in component], dtype=np.float32),
                np.array(component[-1][1:], dtype=np.float32),
            ],
            dtype=np.float32,
        )
        # R2 Score
        numerator = ((y - cache) ** 2).sum(axis=0, dtype=np.float32)
        denominator = ((y - np.average(y, axis=0)) ** 2).sum(axis=0, dtype=np.float32)
        r2[name] = 1 - (numerator / denominator)
    return r2


def evaluate_modularprophet(
    dataset,
    n_forecasts,
    n_lags,
    step_size,
    holdout,
    seasonal_cycle,
    model=None,
    constraint=None,
    n_rows=-1,
):
    df = pd.read_csv(dataset).tail(n_rows)
    num_splits = (holdout - 2 * n_forecasts) + 1
    metrics = {}
    predictions = []
    attributions = {}

    for i in range(0, num_splits, step_size):
        print(f"Split {i} of {math.ceil(num_splits/ step_size)}")
        train_df, eval_df, y = split_data(df, i, holdout, n_forecasts, n_lags)
        forecaster = ModularProphet(model)
        _ = forecaster.fit(
            train_df, n_forecasts=n_forecasts, learning_rate=0.1, constraint=constraint
        )
        # Evaluation
        _, _, y_hat, _ = forecaster.predict(eval_df)
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
        # Calculate component attributions
        _, _, _, components = forecaster.predict(train_df)
        y = train_df["y"].values
        attributions[i] = calculate_component_attributions(components, y, n_lags)
    return (
        pd.DataFrame(metrics).T,
        pd.DataFrame(np.array(predictions)).T,
        pd.DataFrame(attributions).T,
    )
