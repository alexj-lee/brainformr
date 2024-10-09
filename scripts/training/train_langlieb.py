import pathlib
import logging

import anndata as ad
import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
from lightning_model import get_timestamp
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from train_aibs_mouse import AIBSTrainer

from brainformr.data import CenterMaskSampler, collate

# from brainformr import __version__ as brainformr_version
brainformr_version = "1.0"

def load_data(self, config: DictConfig, inference: bool = False):

    data_root = pathlib.Path(config.data.adata_path)

    h5_dir = data_root / "h5ad_normM_log1p"
    metadata_dir = data_root / "mapping"
    trn_samplers = []
    valid_samplers = []
    glob = h5_dir.glob("Puck_Num_*.h5ad")
    glob = sorted(glob, key=lambda el: int(el.name.replace('.h5ad', '').split('_')[-1]))

    for h5_path in glob:#h5_dir.glob("Puck_Num_*.h5ad"):
        adata = ad.read_h5ad(h5_path)

        puck_num = h5_path.stem

        metadata_path = metadata_dir / f"{puck_num}.mapping_metadata.csv"
        metadata = pd.read_csv(metadata_path)

        adata = adata[metadata['cell_label']]
        #adata.X = adata.X.log1p()
        # log xfm nonzero

        assert np.isnan(adata.X.data).sum() == 0, f"Detected a nan while loading {h5_path}"

        metadata.rename(columns={'Raw_Slideseq_X': 'x',
                                 'Raw_Slideseq_Y': 'y',
                                 'PuckID': 'brain_section_label',
                                 }, inplace=True)
        
        metadata['cell_type'] = metadata['cell_index'].astype(int)
        metadata = metadata.reset_index(drop=True)

        metadata = metadata[["cell_type", 'cell_label', 'x', 'y', 'brain_section_label']]


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
        logging.info(f"Loaded {puck_num} with {len(train_indices)} training cells and {len(valid_indices)} validation cells. Number of genes is {adata.shape[1]}")

        trn_samplers.append(train_sampler)
        valid_samplers.append(valid_sampler)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(trn_samplers),
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=False,
        collate_fn=collate,
        prefetch_factor=4,
    )

    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(valid_samplers),
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=False,
        collate_fn=collate,
        prefetch_factor=4,
    )

    return train_loader, valid_loader
    
      
AIBSTrainer.load_data = load_data

@hydra.main(
    config_path="/home/ajl/work/d2/code/brainformr/scripts/config",
    config_name="langlieb.yaml",
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

    print(lightning_model)
       
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
        save_top_k=10,
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
        #   auto_wrap_policy={nn.TransformerEncoderLayer}
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
