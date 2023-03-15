from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Component(ABC, nn.Module):
    def __init__(self, name, feature, id=None):
        super().__init__()
        self.name = name
        self.feature = feature
        self.n_forecasts = None
        self.id = id

    def post_init(self, n_forecasts):
        self.n_forecasts = n_forecasts

    @abstractmethod
    def forward(self, x):
        pass

    def freeze(self):
        pass

    def __repr__(self):
        return f"{self.name}{tuple(self.kwargs.values())}"


class Trend(Component):
    def __init__(self, feature="time"):
        super().__init__("Trend", feature)
        self.kwargs = locals()
        self.kwargs.pop("self")
        self.kwargs.pop("__class__")

        self.trend_k0, self.trend_k1 = self.configure_model()
        self.stationary = False

    def configure_model(self):
        trend_k0 = nn.Parameter(
            nn.init.xavier_normal_(torch.randn(1, 1)), requires_grad=True
        )
        trend_k1 = nn.Parameter(
            nn.init.xavier_normal_(torch.randn(1, 1)), requires_grad=True
        )
        return trend_k0, trend_k1

    def forward(self, x):
        return self.trend_k0 + self.trend_k1 * x[self.feature]


class Regressor(Component):
    def __init__(self, id="regressor_01"):
        super().__init__("Regressor", id=id)
        self.kwargs = locals()
        self.kwargs.pop("self")
        self.kwargs.pop("__class__")

        self.id = id

    def forward(self, x):
        pass
