import numpy as np
import torch
import torch.nn as nn
from modularprophet.components import Component


# class EncodingSeasonality(Component):
#     def __init__(self, season_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.kwargs = locals()
#         self.kwargs.pop("self")
#         self.kwargs.pop("__class__")
#         self.season_dim = season_dim
#         self.weight = self.configure_model()
#         self.stationary = False

#     def configure_model(self):
#         seasonality_weights = nn.Parameter(
#             nn.init.xavier_normal_(torch.randn(1, self.season_dim)), requires_grad=True
#         )
#         return seasonality_weights

#     def forward(self, x):
#         return torch.sum(
#             self.weight
#             * nn.functional.one_hot(x[self.feature], num_classes=self.season_dim),
#             axis=2,
#         )

#     def freeze(self, neutralize=False):
#         self.weight.requires_grad = False

#     def unfreeze(self, init=False):
#         self.weight.requires_grad = True


class FourierSeasonality(Component):
    def __init__(self, feature, period, series_order, multiply_with=None, growth=None):
        super().__init__("FourierSeasonality", feature)
        self.kwargs = locals()
        self.kwargs.pop("self")
        self.kwargs.pop("__class__")

        self.period = period
        self.series_order = series_order
        self.n_modelled_seasonalities = series_order * 2
        self.multiply_with = multiply_with
        self.growth = growth
        self.stationary = False

        self.weights = self.configure_model(self.n_modelled_seasonalities)
        if self.growth is not None:
            self.shift, self.scale = self.configure_growth()

    def configure_model(self, n_modelled_seasonalities):
        seasonality_weights = nn.Parameter(
            nn.init.xavier_normal_(torch.randn(1, 1, n_modelled_seasonalities)),
            requires_grad=True,
        )
        return seasonality_weights

    def configure_growth(self):
        shift = nn.Parameter(
            nn.init.xavier_normal_(torch.randn(1, 1)), requires_grad=True
        )
        scale = nn.Parameter(
            nn.init.xavier_normal_(torch.randn(1, 1)), requires_grad=True
        )
        return shift, scale

    def forward(self, x):
        if self.growth is not None:
            growth = self.shift + self.scale * x["time"]
            return torch.mul(
                growth,
                torch.sum(
                    torch.mul(self.weights, x[self.feature]),
                    axis=2,
                ),
            )
        return torch.sum(
            torch.mul(self.weights, x[self.feature]),
            axis=2,
        )

    def freeze(self, neutralize=False):
        self.weights.requires_grad = False

    def unfreeze(self, init=False):
        self.weights.requires_grad = True

    def extract_features(self, df):
        """
        Extracts the Fourier seasonality features from the time series.
        """
        df[self.feature] = df["time"].apply(
            lambda t: [
                fun((2.0 * (i + 1) * np.pi * (t / 3600 / 24) / self.period))
                for i in range(self.series_order)
                for fun in (np.sin, np.cos)
            ]
        )
        return df
