import logging
import time
import pandas as pd
import torch.nn as nn

from modularprophet.compositions import Composition, Single
from modularprophet.components import Component
from modularprophet.utils import models_to_summary, validate_inputs
from modularprophet.lightning import LightningModel
from modularprophet.utils_lightning import configure_trainer, find_lr

logger = logging.getLogger("experiments")


class Container(nn.ModuleList):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def _train(
        self,
        model,
        config,
        datamodule,
        experiment_name,
        compute_components=True,
        target=None,
    ):
        """
        Train the provided torch model using Lightning.
        """
        # Wrap model in LightningModule
        model = LightningModel(
            model=model,
            compute_components=compute_components,
            optimizer=config.get("training.optimizer"),
        )

        # Specify the target to train the model on (if specified)
        if target is not None:
            dataloader = datamodule.train_dataloader(target=target)
        else:
            dataloader = datamodule

        #### Trainer ####
        trainer, checkpoint_callback, metrics_logger = configure_trainer(
            config.get_config("training"),
            experiment_name=experiment_name,
        )
        if "learning_rate" in config.get("training").keys():
            model.lr = config.get("training.learning_rate")
        else:
            model.lr = find_lr(trainer, model, dataloader)
            logger.info(f"Found optimal learning rate: {round(model.lr, 6)}")
        #### Train ####
        logger.info("Starting training")

        # Train the model
        t0 = time.time()
        trainer.fit(model, dataloader)
        t1 = time.time()
        logger.info(f"Training took {round(t1 - t0, 2)} seconds")

        # Load best model
        if checkpoint_callback.best_model_score is not None:
            if checkpoint_callback.best_model_score < checkpoint_callback.current_score:
                model = LightningModel.load_from_checkpoint(
                    checkpoint_callback.best_model_path
                )

        metrics = pd.DataFrame(metrics_logger.get_last_metrics())

        return model.model, trainer, metrics

    def fit(self):
        pass

    def post_init(self, n_forecasts):
        for model in self.models:
            if isinstance(model, Component):
                model.post_init(n_forecasts)
            else:
                for m in model:
                    m.post_init(n_forecasts)

    def __repr__(self):
        return models_to_summary(self.name, self.models)


class Model(Container):
    def __init__(self, model):
        super().__init__("Model")
        validate_inputs([model], [Composition, Component])
        if isinstance(model, Component):
            model = Single(model)
        self.models = model
        self.datamodule = None
        self.trainer = None

    def fit(self, config, datamodule, n_forecasts, experiment_name, target):
        self.post_init(n_forecasts=n_forecasts)
        self.datamodule = datamodule
        self.models, self.trainer, metrics = self._train(
            self.models, config, self.datamodule, experiment_name, target
        )
        return metrics

    def predict(self, datamodule):
        self.datamodule = datamodule
        model = LightningModel(self.models)
        predictions_raw = self.trainer.predict(model, self.datamodule)
        return predictions_raw


class Sequential(Container):
    def __init__(self, *models):
        super().__init__("Sequential")
        validate_inputs(models, [Composition, Component])
        self.models = models

    def fit(self, df):
        for model in self.models:
            model.fit(df)
            prediction = model.predict(df)
            df = df - prediction
        return model


class Ensemble(Container):
    def __init__(self, *models):
        super().__init__("Ensemble")
        validate_inputs(models, [Composition, Component])
        self.models = models

    def fit(self, df):
        models = []
        for model in self.models:
            models[model.name] = model.fit(df)
        return models

    def predict(self, df):
        predictions = []
        for model in self.models:
            predictions[model.name] = model.predict(df)
        return predictions.avg()
