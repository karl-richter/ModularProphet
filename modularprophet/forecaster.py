import logging
import numpy as np
import pandas as pd

from modularprophet.containers import Container, Model
from modularprophet.compositions import Composition
from modularprophet.components import Component
from modularprophet.dataset import TimeDataModule
from modularprophet.utils import get_n_lags_from_model

logger = logging.getLogger("experiments")

# https://github.com/amarczew/pytorch_model_summary


class ModularProphet:
    def __init__(self, model):
        self.model = self.validate_model(model)
        self.experiment_name = None
        self.n_forecasts = None

    def validate_model(self, model):
        if isinstance(model, Container):
            return model
        elif isinstance(model, Composition) or isinstance(model, Component):
            return Model(model)
        else:
            raise TypeError(
                f"The model must of type Container, Model or Component. The type provided is: {str(type(model))}."
            )

    def fit(
        self,
        df: pd.DataFrame,
        n_forecasts,
        optimizer="bfgs",
        learning_rate=None,
        epochs=50,
        batch_size=128,
    ):
        n_lags = get_n_lags_from_model(self.model)
        self.n_forecasts = n_forecasts
        self.datamodule = TimeDataModule(
            df, self.model, n_forecasts, n_lags, optimizer, batch_size
        )
        metrics = self.model.fit(
            self.datamodule,
            n_forecasts,
            optimizer,
            learning_rate,
            epochs,
            self.experiment_name,
            None,
        )
        return metrics

    def denormalize(self, array, kind="target"):
        return array * self.datamodule.scale[kind] + self.datamodule.shift[kind]

    def predict(self, df: pd.DataFrame):
        self.datamodule.update_predict_df(df)
        predictions_raw = self.model.predict(self.datamodule)
        # if predictions_raw[0]["components"] is None:
        #     predictions_raw[0]["components"] = {}
        # Convert batches of torch tensors to numpy arrays
        x = self.denormalize(
            np.concatenate([batch["x"].numpy() for batch in predictions_raw], axis=0),
            kind="time",
        )
        y = self.denormalize(
            np.concatenate([batch["y"].numpy() for batch in predictions_raw], axis=0)
        )
        y_hat = self.denormalize(
            np.concatenate(
                [batch["y_hat"].numpy() for batch in predictions_raw], axis=0
            )
        )
        components = {
            component: self.denormalize(
                np.concatenate(
                    [batch["components"][component] for batch in predictions_raw],
                    axis=0,
                )
            )
            for component in predictions_raw[0]["components"].keys()
        }
        components["y_hat"] = y_hat
        return x, y, y_hat, components

    def __repr__(self):
        return self.model.__repr__()
