import logging
import pandas as pd

from modularprophet.containers import Container, Model
from modularprophet.compositions import Composition
from modularprophet.components import Component

logger = logging.getLogger("experiments")

# https://github.com/amarczew/pytorch_model_summary


class ModularProphet:
    def __init__(self, model):
        self.model = self.validate_model(model)
        self.datamodule = None
        self.experiment_name = None
        self.target = None

    def validate_model(self, model):
        if isinstance(model, Container):
            return model
        elif isinstance(model, Composition) or isinstance(model, Component):
            return Model(model)
        else:
            raise TypeError(
                f"The model must of type Container, Model or Component. The type provided is: {str(type(model))}."
            )

    def fit(self, df: pd.DataFrame, config):
        self.datamodule = df
        metrics = self.model.fit(config, self.datamodule, self.experiment_name, None)
        return metrics

    def predict(self, df: pd.DataFrame):
        predictions = self.trainer.predict(self.model.model, self.datamodule)
        return predictions

    def __repr__(self):
        return self.model.__repr__()
