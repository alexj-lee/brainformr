import pathlib
import warnings

import anndata as ad
import hydra
import lightning as L
import pandas as pd
import torch
import wandb
from lightning_model import get_timestamp
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch import nn
from train_aibs_mouse import AIBSTrainer

from brainformr.data import CenterMaskSampler, collate

# from brainformr import __version__ as brainformr_version
brainformr_version = "1.0"


def load_data(self, config: DictConfig, inference: bool = False):
    adata = ad.read_h5ad(config.data.adata_path)
    # filter control probes
    adata[:, ~adata.var.index.str.contains("Blank")] 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        metadata = pd.read_csv(config.data.metadata_path)

    metadata["cell_label"] = metadata["cell_label"].astype(str)
    metadata["x"] = metadata["spatial_x"] / 10.
    metadata["y"] = metadata["spatial_y"] / 10.

    metadata = metadata[
        [
            config.data.celltype_colname,
            "cell_label",
            "x",
            "y",
            "brain_section_label",
        ]
    ]

    metadata["cell_type"] = self.label_to_cls(
        metadata[config.data.celltype_colname]
    )
    metadata["cell_type"] = metadata["cell_type"].astype(int)

    metadata = metadata.reset_index(drop=True)

    # filter for MERFISH cells that were clf'd
    adata = adata[metadata["cell_label"]]

    if inference:
        sampler = CenterMaskSampler(
            metadata=metadata,
            adata=adata,
            patch_size=config.data.patch_size,
            cell_id_colname=config.data.cell_id_colname,
            cell_type_colname="cell_type",
            tissue_section_colname=config.data.tissue_section_colname,
            max_num_cells=config.data.neighborhood_max_num_cells,
        )

        loader = torch.utils.data.DataLoader(
            sampler,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            collate_fn=collate,
            prefetch_factor=4,
        )

        return loader
    else:
        train_indices, valid_indices = train_test_split(
            range(len(adata)), train_size=config.data.train_pct
        )

        train_sampler = CenterMaskSampler(
            metadata=metadata,
            adata=adata,
            patch_size=config.data.patch_size,
            cell_id_colname=config.data.cell_id_colname,
            cell_type_colname="cell_type",
            tissue_section_colname=config.data.tissue_section_colname,
            max_num_cells=config.data.neighborhood_max_num_cells,
            indices=train_indices,
        )

        valid_sampler = CenterMaskSampler(
            metadata=metadata,
            adata=adata,
            patch_size=config.data.patch_size,
            cell_id_colname=config.data.cell_id_colname,
            cell_type_colname="cell_type",
            tissue_section_colname=config.data.tissue_section_colname,
            max_num_cells=config.data.neighborhood_max_num_cells,
            indices=valid_indices,
        )

        train_loader = torch.utils.data.DataLoader(
            train_sampler,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            collate_fn=collate,
            prefetch_factor=4,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_sampler,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True,
            collate_fn=collate,
            prefetch_factor=4,
        )

        return train_loader, valid_loader
      
AIBSTrainer.load_data = load_data

@hydra.main(
    config_path="/home/ajl/work/d2/code/brainformr/scripts/config",
    config_name="mouse_687997.yaml",
)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    L.seed_everything(1221)
    torch.set_float32_matmul_precision("high")

    setup_training(config)


def setup_training(config: DictConfig):
    timestamp = get_timestamp()

    lightning_model = AIBSTrainer(config)
    trn_loader, valid_loader = lightning_model.load_data(config)

    if config.model_checkpoint not in (None, '', 'None'):
       lightning_model.load_checkpoint(config.model_checkpoint, lightning=True)
       
    lightning_model.compile_specific()

    checkpoint_dir_root = pathlib.Path(config.checkpoint_dir) / timestamp
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict["version"] = brainformr_version
    config_dict["checkpoint_dir_root"] = checkpoint_dir_root

    (checkpoint_dir_root / "wandb").mkdir(exist_ok=True, parents=True)
    with open(checkpoint_dir_root / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    logger = L.pytorch.loggers.wandb.WandbLogger(
        project=config.wandb_project,
        save_dir=checkpoint_dir_root,
        config=wandb.helper.parse_config(config_dict),
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
    logger.watch(lightning_model, log="gradients")

    chkpoint = L.pytorch.callbacks.ModelCheckpoint(
        monitor="valid/nll_epoch",
        dirpath=checkpoint_dir_root,
        filename="model-{epoch:02d}-{validNLL_:.2f}",
        save_top_k=2,
        mode="min",
    )

    trainer = L.Trainer(
        log_every_n_steps=100,
        max_epochs=config.optimization.epochs,
        accumulate_grad_batches=config.optimization.accumulation_steps,
        precision=config.optimization.precision,
        accelerator="cuda",
        devices=[0, 1],
        callbacks=[lr_monitor, chkpoint],
        strategy=L.pytorch.strategies.FSDPStrategy(
            #auto_wrap_policy={nn.TransformerEncoderLayer}
        ),
        logger=logger,
        deterministic=True,
    )

    trainer.fit(
        lightning_model,
        trn_loader,
        valid_loader,
    )

    return timestamp


if __name__ == "__main__":
    main()
