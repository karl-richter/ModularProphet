import logging
import time
import pandas as pd

from modularprophet.compositions import Composition
from modularprophet.components import Component
from modularprophet.utils import models_to_summary, validate_inputs
from modularprophet.lightning import LightningModel

logger = logging.getLogger("experiments")


class Container:
    def __init__(self, name):
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
            metrics=config.get("metrics"),
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
            model.lr, fig_lr = find_lr(trainer, model, dataloader)
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

        return model, metrics

    def fit(self):
        pass

    def __repr__(self):
        return models_to_summary(self.name, self.models)


class Model(Container):
    def __init__(self, model):
        super().__init__("Model")
        validate_inputs([model], [Composition, Component])
        self.models = model

    def fit(self, config, dataloader, experiment_name, target):
        self.model, metrics = self._train(
            self.model, config, dataloader, experiment_name, target
        )
        return metrics


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
