from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional

import lightning as L
import torch
import wandb
from hydra.utils import instantiate
from jaxtyping import Float
from omegaconf import DictConfig
from scvi.distributions import ZeroInflatedNegativeBinomial
from torch import optim

from brainformr.analysis_utils.crosscorr import corr_predictions
from brainformr.training.scheduler import get_inverse_sqrt_schedule_with_plateau


def get_timestamp():
    return str(datetime.today().strftime("%Y-%m-%d-%H%S"))

class BaseTrainer(L.LightningModule, ABC):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()

        model = instantiate(config.model)
        self.model = model
        self.global_config = config

    def forward(self, x):
        return self.model(x)

    def _step_zinb(
        self,
        zinb_obj: ZeroInflatedNegativeBinomial,
        expression: Float[torch.Tensor, "n_cells n_genes"],  # noqa: F722
        return_expression: Optional[bool] = False,
    ):
        h_expr_counts = torch.pow(expression, 2)
        loss = -zinb_obj.log_prob(h_expr_counts)

        if return_expression:
            return loss, h_expr_counts
        return loss

    def training_step(self, data_dict: dict):
        fwd = self.forward(data_dict)
        zinb_params: Dict[str, Float[torch.Tensor, "n_cells n_genes"]] = fwd[  # noqa: F722
            "zinb_params"
        ]

        zinb = ZeroInflatedNegativeBinomial(**zinb_params)
        loss = self._step_zinb(zinb, data_dict["hidden_expression"])

        self.log(
            "train/nll",
            loss,
            batch_size=data_dict["bs"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
        )

    def compute_cross_corr(
        self,
        zinb_obj: ZeroInflatedNegativeBinomial,
        expression: Float[torch.Tensor, "n_cells n_genes"],  # noqa: F722
    ):
        average_expr = zinb_obj.mean
        cross_corr = corr_predictions(expression, average_expr)
        return cross_corr.mean()

    def validation_step(self, data_dict: dict):
        fwd = self.forward(data_dict)
        zinb_params: Dict[str, Float[torch.Tensor, "n_cells n_genes"]] = fwd[  # noqa: F722
            "zinb_params"
        ]

        zinb = ZeroInflatedNegativeBinomial(**zinb_params)
        loss, expr_counts = self._step_zinb(
            zinb, data_dict["hidden_expression"], return_expression=True
        )
        cross_corr = self.compute_cross_corr(zinb, expr_counts)

        self.log_dict(
            {"valid/nll": loss.item(), "valid/crosscorr": cross_corr},
            batch_size=data_dict["bs"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
        )

    def on_train_start(self):
        if wandb.run and (
            self.global_config.wandb_code_dir is not None
            and self.global_config.wandb_code_dir != ""
        ):
            wandb.run.log_code(
                root=self.global_config.wandb_code_dir,
                include_fn=lambda path: path.endswith(".py"),
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            lr=self.global_config.optimization.lr,
            weight_decay=self.global_config.optimization.wd,
            params=self.parameters(),
        )

        scheduler = get_inverse_sqrt_schedule_with_plateau(
            optimizer,
            self.global_config.optimization.schedule_warmup,
            timescale=self.global_config.optimization.schedule_timescale,
            plateau_length=self.global_config.optimization.schedule_plateau,
        )

        return [optimizer], dict(scheduler=scheduler, frequency=1, interval="step")

    @abstractmethod
    def load_data(self):
        pass
