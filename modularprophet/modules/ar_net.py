import torch
import torch.nn as nn
from modularprophet.components import Component


class LaggedNet(Component):
    def __init__(self, n_lags, n_hidden=0, d_hidden=0, feature="lags"):
        super().__init__("LaggedNet", feature)
        self.kwargs = locals()
        self.kwargs.pop("self")
        self.kwargs.pop("__class__")
        self.n_lags = n_lags
        self.n_hidden = n_hidden
        self.d_hidden = d_hidden
        self.network = None

    def post_init(self, n_forecasts):
        super().post_init(n_forecasts)
        self.network = self.configure_model(
            d_inputs=self.n_lags,
            n_hidden=self.n_hidden,
            d_hidden=self.d_hidden,
            d_outputs=self.n_forecasts,
        )

    def configure_model(self, d_inputs, d_hidden, n_hidden, d_outputs):
        layers = []
        for i in range(n_hidden):
            layers.append(nn.Linear(d_inputs, d_hidden, bias=True))
            layers.append(nn.ReLU())
            d_inputs = d_hidden
        layers.append(nn.Linear(d_inputs, d_outputs, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x[self.feature])

    def freeze(self, neutralize=False):
        for param in self.network.parameters():
            param.requires_grad = False
            if neutralize:
                param.data = torch.zeros_like(param.data)

    def unfreeze(self, init=False, init_weights=None):
        for param in self.network.parameters():
            param.requires_grad = True
            if init:
                if init_weights is not None:
                    param.data = torch.tensor(
                        init_weights, dtype=param.data.dtype
                    ).reshape(param.data.shape)
                else:
                    nn.init.xavier_normal_(torch.randn_like(param.data))
