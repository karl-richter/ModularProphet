# ModularProphet

ModularProphet is a modular hybrid forecasting framework for interpretable time series forecasting. ModularProphet is re-designed and extendable version of [NeuralProphet](https://github.com/ourownstory/neural_prophet). The framework is built on PyTorch and combines Neural Network and traditional time-series algorithms, inspired by Facebook Prophet and AR-Net.

The framework allows to combine components arbitrarily. An example can be found in the following:
```python
m = ModularProphet(
    Stationary(
        Trend(),
        FourierSeasonality("yearly", period=365.25, series_order=5, growth="linear"),
        FourierSeasonality("monthly", period=30.5, series_order=5, growth="linear"),
        FourierSeasonality("weekly", period=7, series_order=5),
        FourierSeasonality("daily", period=1, series_order=5),
        LaggedNet(n_lags=168)
    ),
)
```

## About
This framework has been developed as part of the Master thesis "Interpretability Through Modularity: A Modular Framework for Hybrid Forecasting Model Creation" during a research stay at the Stanford Sustainable Systems Lab (S3L).

## Cite
Please cite ModularProphet in your publications if it helps your research:

```
@software{richter2023modularprophet,
  author = {Richter, Karl},
  title = {{ModularProphet}},
  url = {https://github.com/karl-richter/ModularProphet},
  month = {04},
  year = {2023}
}
```
