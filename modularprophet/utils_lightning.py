import os
import numpy as np
import collections
from typing import Any, Mapping, Optional

from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


def configure_trainer(config, experiment_name: str) -> Trainer:
    #### Callbacks ####
    checkpoint_callback = ModelCheckpoint(
        monitor=config.get("optimization_metric", "loss"),
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    # TODO: configure early stopping
    # early_stop_callback = EarlyStopping(
    #     monitor="loss", mode="min", patience=20, divergence_threshold=20.0
    # )

    #### Logger ####
    metrics_logger = MetricsLogger(save_dir="logs", name=experiment_name)

    #### Progress bar ####

    progress_bar_callback = LightningProgressBar(
        refresh_rate=50, max_epochs=config.get("epochs")
    )

    #### Trainer ####
    trainer = Trainer(
        # accelerator="mps",
        # devices=1,
        max_epochs=config.get("epochs") if config.get("optimizer") != "bfgs" else 1,
        limit_train_batches=config.get("limit_train_batches", None),
        logger=metrics_logger,
        callbacks=[],  # checkpoint_callback, progress_bar_callback
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=True,
        replace_sampler_ddp=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    return trainer, checkpoint_callback, metrics_logger


def find_lr(trainer, net, train_loader):
    """
    Find the optimal learning rate for the model.

    Note: The default learning rate distribution is not smoothed by default, thus the selected learning rate is usually
    far off. We use a hamming filter to smooth the learning rate distribution and only then select a lr using the
    steepest gradient.
    """
    lr_finder = trainer.tuner.lr_find(
        net,
        train_dataloaders=train_loader,
        min_lr=1e-6,
        max_lr=10,
        num_training=200,
        early_stop_threshold=None,
        mode="exponential",
    )

    # Smooth the loss using a hamming filter
    loss = np.pad(np.array(lr_finder.results["loss"]), pad_width=15, mode="edge")
    window = np.hamming(30)
    loss = np.convolve(
        window / window.sum(),
        loss,
        mode="valid",
    )[1:]
    # Find the steepest gradient and the minimum loss after that
    steepest_gradient_idx = np.argmin(np.gradient(loss))
    min_loss_idx = np.argmin(loss[steepest_gradient_idx:])
    # Select the average of the two (more conservative than just using the steepest gradient)
    loss_idx = steepest_gradient_idx + int(min_loss_idx / 2)
    lr = lr_finder.results["lr"][loss_idx]

    # Remove uneeded learning rate finder checkpoints
    for f in os.listdir():
        if ".lr_find_" in f:
            os.remove(f)

    return lr


class LightningProgressBar(TQDMProgressBar):
    """
    Custom progress bar for PyTorch Lightning for only update every epoch, not every batch.
    """

    def __init__(
        self,
        **kwargs: Any,
    ):
        self.max_epochs = kwargs.pop("max_epochs")
        super().__init__(**kwargs)

    def on_train_epoch_start(self, trainer: "Trainer", *_) -> None:
        self.main_progress_bar.reset(self.max_epochs)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch + 1}")
        self._update_n(self.main_progress_bar, trainer.current_epoch + 1)

    def on_train_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", *_
    ) -> None:
        pass

    def _update_n(self, bar, value: int) -> None:
        if not bar.disable:
            bar.n = value
            bar.refresh()


class MetricsLogger(TensorBoardLogger):
    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.history = collections.defaultdict(list)
        self.checkpoint_path = None
        self.latest_metrics = None

    def after_save_checkpoint(self, checkpoint_callback) -> None:
        """Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance
        """
        self.checkpoint_path = checkpoint_callback.best_model_path

    @rank_zero_only
    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
        super(MetricsLogger, self).log_metrics(metrics, step)
        # metrics is a dictionary of metric names and values
        if not len(self.history["epoch"]) or not self.history["epoch"][-1] == (
            metrics["epoch"] + 1
        ):
            for metric_name, metric_value in metrics.items():
                if metric_name == "hp_metric":
                    pass
                elif metric_name == "epoch":
                    self.history[metric_name].append(metric_value + 1)
                else:
                    self.history[metric_name].append(metric_value)
        else:
            self.latest_metrics = metrics
        return

    def get_last_metrics(self):
        if self.latest_metrics is not None:
            for metric_name, metric_value in self.latest_metrics.items():
                if metric_name == "hp_metric":
                    pass
                elif metric_name == "epoch":
                    self.history[metric_name].append("last")
                else:
                    self.history[metric_name].append(metric_value)
            # Reset the latest metrics
            self.latest_metrics = None
        return self.history
