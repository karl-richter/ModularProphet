from abc import ABC, abstractmethod


class Component(ABC):
    def __init__(self, name, id=None):
        self.name = name
        self.id = id

    @abstractmethod
    def forward(self, x):
        pass

    def freeze(self):
        pass

    def __repr__(self):
        return f"{self.name}{tuple(self.kwargs.values())}"


class Trend(Component):
    def __init__(
        self,
        growth="linear",
        changepoints=None,
        n_changepoints: int = 10,
        changepoints_range: float = 0.8,
        trend_reg: float = 0,
        trend_reg_threshold=False,
        trend_global_local="global",
    ):
        super().__init__("Trend")
        self.kwargs = locals()
        self.kwargs.pop("self")
        self.kwargs.pop("__class__")

        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoints_range = changepoints_range
        self.trend_reg = trend_reg
        self.trend_reg_threshold = trend_reg_threshold
        self.trend_global_local = trend_global_local
    
    def forward(self, x):
        pass


class Regressor(Component):
    def __init__(self, id="regressor_01"):
        super().__init__("Regressor", id=id)
        self.kwargs = locals()
        self.kwargs.pop("self")
        self.kwargs.pop("__class__")

        self.id = id
    
    def forward(self, x):
        pass
