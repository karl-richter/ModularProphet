import logging
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler


logger = logging.getLogger("experiments")


class LightningModel(pl.LightningModule):
    """
    Lightning wrapper to optimize the weights of a model.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        constraint=None,
        compute_components=True,
        **kwargs,
    ):
        super().__init__()
        self.lr = 0.001
        self.model = model
        self._optimizer = optimizer
        self.constraint = constraint
        self.compute_components = compute_components

    def configure_optimizers(self):
        if self._optimizer == "adam":
            self.optimizer = optim.AdamW(
                self.parameters(), weight_decay=1e-5, lr=self.lr
            )
            self.scheduler = {
                "scheduler": lr_scheduler.OneCycleLR(
                    self.optimizer,
                    **{
                        "pct_start": 0.3,
                        "anneal_strategy": "cos",
                        "div_factor": 200.0,
                        # "final_div_factor": 5000.0,
                        "max_lr": self.lr,
                        "total_steps": self.trainer.max_epochs
                        * self.trainer.estimated_stepping_batches,
                    },
                ),
                "name": "lr",
                "interval": "step",
            }
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
        elif self._optimizer == "bfgs":
            self.optimizer = torch.optim.LBFGS(
                self.parameters(), max_iter=50, lr=self.lr
            )
            return self.optimizer
        elif self._optimizer is not None:
            raise ValueError(f"Invalid optimizer: {self._optimizer}")

    def loss_func(self, y_hat, y):
        return nn.SmoothL1Loss(reduction="mean")(y_hat, y)

    def constrain_stationarity(self, components, y):
        stationary_components = [c for c in components.keys() if c != "LaggedNet"]
        return self.loss_func(
            torch.stack([components[c] for c in stationary_components], axis=0).sum(
                axis=0
            ),
            y["target"],
        )

    def constrain_incremental(self, components, y):
        component_loss = {}
        prev_components = None
        for name, component in components.items():
            component_loss[name] = self.loss_func(component, y["target"])
            if prev_components is not None:
                prev_components = torch.stack([prev_components, component], axis=0).sum(
                    axis=0
                )
            else:
                prev_components = component
        return torch.stack([*component_loss.values()], axis=0).sum(axis=0) * 0.1

    def constrain_zero_mean(self, components, y):
        component_loss = {}
        for component in [c for c in self.model.components if c.name == "LaggedNet"]:
            # Calculate the sum of the absolute integral between the y and x axis
            component_loss[component.name] = torch.sum(components[component.name])
        return torch.stack([*component_loss.values()], axis=0).sum(axis=0) * 0.1

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, components = self.model.forward(x)
        loss = self.loss_func(y_hat, y["target"])
        # Adding constraints
        if components is not None:
            if self.constraint == "stationarity":
                loss += self.constrain_stationarity(components=components, y=y)
            elif self.constraint == "incremental":
                loss += self.constrain_incremental(components=components, y=y)
            elif self.constraint == "zeromean":
                loss += self.constrain_zero_mean(components=components, y=y)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat, components = self.model.forward(x)
        return {
            "x": x["time"],
            "y": y["target"],
            "y_hat": y_hat,
            "components": components,
        }
