from abc import ABC, abstractmethod
from modularprophet.components import Component
from modularprophet.utils import components_to_summary, validate_inputs


class Composition(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def forward(self, x):
        pass

    def __repr__(self):
        return components_to_summary(self.name, self.components)


class Additive(Composition):
    def __init__(self, *components):
        super().__init__("Additive")
        validate_inputs(components, Component)
        self.components = components

    def forward(self, x):
        components = {}
        for component in self.components:
            components[component.name] = component.forward(x)
        prediction = sum(components.values())
        return prediction, components


class Stationary(Composition):
    def __init__(self, *components):
        super().__init__("Stationary")
        validate_inputs(components, Component)
        self.components = components

    def forward(self, x):
        non_stationary_components = {}
        for component in [c for c in self.components if not c.stationary]:
            non_stationary_components[component.name] = component.forward(x)

        x["lags"] = x["lags"] - sum(non_stationary_components.values())

        components = {}
        for component in self.components:
            components[component.name] = component.forward(x)
        prediction = sum(components.values())
        return prediction, components


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