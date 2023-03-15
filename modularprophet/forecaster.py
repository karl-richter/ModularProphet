import logging
import numpy as np
import pandas as pd

from modularprophet.containers import Container, Model
from modularprophet.compositions import Composition
from modularprophet.components import Component
from modularprophet.dataset import TimeDataModule

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

    def fit(self, df: pd.DataFrame, config, n_forecasts, batch_size=128):
        self.datamodule = TimeDataModule(df, config, n_forecasts, batch_size)
        metrics = self.model.fit(
            config, self.datamodule, n_forecasts, self.experiment_name, None
        )
        return metrics

    def predict(self, df: pd.DataFrame):
        predictions_raw = self.model.predict()
        if predictions_raw[0]["components"] is None:
            predictions_raw[0]["components"] = {}
        # Extract 1-step ahead forecast from batches
        predictions_list = [
            {
                "x": batch["x"][:, 0].detach().numpy(),
                "y": batch["y"][:, 0].detach().numpy(),
                "y_hat": batch["y_hat"][:, 0].detach().numpy(),
                **{
                    name: c[:, 0].detach().numpy()
                    for name, c in batch["components"].items()
                },
            }
            for batch in predictions_raw
        ]

        # Append the n-step ahead forecast of the last batch
        predictions_list.append(
            [
                {
                    "x": batch["x"][-1, 1:].detach().numpy(),
                    "y": batch["y"][-1, 1:].detach().numpy(),
                    "y_hat": batch["y_hat"][-1, 1:].detach().numpy(),
                    **{
                        name: c[-1, 1:].detach().numpy()
                        for name, c in batch["components"].items()
                    },
                }
                for batch in [predictions_raw[-1]]
            ][0]
        )

        cols = predictions_list[0].keys()
        predictions_dict = {}
        for column in cols:
            predictions_dict[column] = np.concatenate(
                [batch[column] for batch in predictions_list]
            )

        predictions = pd.DataFrame(predictions_dict)
        return predictions

    def __repr__(self):
        return self.model.__repr__()
