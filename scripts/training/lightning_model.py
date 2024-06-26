from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional

import lightning as L
import pandas as pd
import torch
import wandb
from hydra.utils import instantiate

# from brainform.model.cell_transformer_nb_encdec import (
#     CellLocationTransformer,
# )
from jaxtyping import Float
from omegaconf import DictConfig
from scvi.distributions import ZeroInflatedNegativeBinomial
from sklearn.preprocessing import LabelEncoder
from torch import optim

from brainformr.analysis_utils.crosscorr import corr_predictions
from brainformr.training.scheduler import get_inverse_sqrt_schedule_with_plateau


def get_timestamp():
    return str(datetime.today().strftime("%Y-%m-%d-%H%S"))


class BaseTrainer(L.LightningModule, ABC):
    def __init__(self, config: DictConfig, checkpoint: str | None = None):
        super().__init__()

        model = instantiate(config.model)
        self.model = model
        self.global_config = config
        if (checkpoint is not None) and (checkpoint != ""):
            self.load_checkpoint(checkpoint)

    def label_to_cls(self, labels_str: pd.Series):
        le = LabelEncoder()
        le.fit(sorted(labels_str.unique()))
        return le.transform(labels_str)

    def forward(self, x):
        return self.model(x)

    def _zinb_loss(
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

    def load_checkpoint(self, checkpoint_path: str, lightning: bool = False):
        strict = True
        checkpoint = torch.load(checkpoint_path)

        print('Started loading state dict.')
        if lightning:
            state_dict = {}
            for key in checkpoint["state_dict"]:
                state_dict[key.replace("model.", "")] = checkpoint["state_dict"][key]
        else:
            state_dict = checkpoint['state_dict']

        self.load_state_dict(state_dict, strict=strict)
        print('Finished loading state dict.')

    def training_step(self, data_dict: dict):
        fwd = self.forward(data_dict)
        zinb_params: Dict[str, Float[torch.Tensor, "n_cells n_genes"]] = fwd[  # noqa: F722
            "zinb_params"
        ]
        # zinb_params = dict(mu=fwd['mu'] + 1e-9, theta=fwd['theta'] + 1e-9, zi_logits=fwd['gate'], scale=fwd['scale'] + 1e-9)

        zinb = ZeroInflatedNegativeBinomial(**zinb_params)
        loss = self._zinb_loss(zinb, fwd["hidden_expression"]).mean()

        self.log(
            "train/nll",
            loss,
            batch_size=data_dict["bs"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        return loss

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
        # zinb_params = dict(mu=fwd['mu'] + 1e-9, theta=fwd['theta'] + 1e-9, zi_logits=fwd['gate'], scale=fwd['scale'] + 1e-9)

        zinb = ZeroInflatedNegativeBinomial(**zinb_params)
        loss, expr_counts = self._zinb_loss(
            zinb, fwd["hidden_expression"], return_expression=True
        )
        loss = loss.mean()  # 2500 steps, 0.9ish  loss

        cross_corr = self.compute_cross_corr(zinb, expr_counts)

        self.log_dict(
            {"valid/nll": loss.item(), "valid/crosscorr": cross_corr},
            batch_size=data_dict["bs"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        return loss

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
