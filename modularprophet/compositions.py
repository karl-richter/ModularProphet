from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from modularprophet.components import Component
from modularprophet.utils import components_to_summary, validate_inputs


class Composition(ABC, nn.ModuleList):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x):
        pass

    def extract_features(self, df):
        for component in self.components:
            df = component.extract_features(df)
        return df

    def get_features(self):
        features = []
        for component in self.components:
            features.extend(component.get_features())
        return features

    def __iter__(self):
        return self.components.__iter__()

    def __repr__(self):
        return components_to_summary(self.name, self.components)


class Single(Composition):
    def __init__(self, *components):
        super().__init__("Single")
        validate_inputs(components, Component)
        assert len(components) == 1
        self.components = nn.ModuleList(components)

    def forward(self, x):
        components = {}
        prediction = self.components[0].forward(x)
        components[self.components[0].name] = prediction
        return prediction, components


class Additive(Composition):
    def __init__(self, *components):
        super().__init__("Additive")
        validate_inputs(components, [Component, Composition])
        self.components = nn.ModuleList(components)

    def forward(self, x):
        components = {}
        for component in self.components:
            forward = component.forward(x)
            if (
                hasattr(component, "multiply_with")
                and component.multiply_with is not None
            ):
                #            if component.multiply_with is not None:
                forward = torch.multiply(forward, components[component.multiply_with])
            if isinstance(component, Composition):
                components.update(forward[1])
            else:
                components[component.name] = forward
        y_hat = torch.stack([*components.values()], axis=0).sum(axis=0)
        return y_hat, components


class Stationary(Composition):
    def __init__(self, *components):
        super().__init__("Stationary")
        validate_inputs(components, Component)
        self.components = nn.ModuleList(components)

    def forward(self, x):
        non_stationary_components = {}
        for component in [c for c in self.components if not c.stationary]:
            non_stationary_forward = component.forward(x["lagged"])
            if component.multiply_with is not None:
                non_stationary_forward = torch.multiply(
                    non_stationary_forward,
                    non_stationary_components[component.multiply_with],
                )
            non_stationary_components[component.name] = non_stationary_forward

        # Sum all non-stationary components
        y_hat_non_stationary = torch.stack(
            [*non_stationary_components.values()], axis=0
        ).sum(axis=0)

        # Subtract portion explained by the non-stationary components from the lags
        # NOTE: This is a critical point in the compoitation, to allow for gradient flow, one is not allowed to
        # overwrite objects that are involved in the computational graph. Thus, the following would break the code.
        #    x["lags"] = x["lags"] - y_hat_non_stationary
        # Instead, we create a new dictionary with the updated values.
        x_ = {k: x[k] for k in x.keys() if k != "lags"}
        x_["lags"] = x["lags"] - y_hat_non_stationary

        # Component-wise forward pass of stationary components
        components = {}
        for component in self.components:
            forward = component.forward(x_)
            if component.multiply_with is not None:
                forward = torch.multiply(forward, components[component.multiply_with])
            # Add the component to the dictionary
            components[component.name] = forward

        # Component composition
        y_hat = torch.stack([*components.values()], axis=0).sum(axis=0)
        return y_hat, components


class Weighted(Composition):
    def __init__(self, *components):
        super().__init__("Weighted")
        validate_inputs(components, Component)
        self.components = components
        self.component_weights = {}

    def forward(self, x):
        components = {}
        for component in self.components:
            components[component.name] = (
                component.forward(x) * self.component_weights[component.name]
            )
        prediction = sum(components.values())
        return prediction, components
