import argparse
import os
import sys
import warnings

import anndata as ad
import pandas as pd
import torch
import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.preprocessing import LabelEncoder

from brainformr.data import CenterMaskSampler, collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata_path", type=str, help="Path to anndata file.")
    parser.add_argument("--metadata_path", type=str, help="Path to metadata file.")
    parser.add_argument("--config_path", type=str, help="Path to config file.")
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint file from training.')
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.adata_path):
        print(f"Anndata file not found: {args.adata_path}")
        sys.exit(1)

    if not os.path.exists(args.metadata_path):
        print(f"Metadata file not found: {args.metadata_path}")
        sys.exit(1)

    adata = ad.read_h5ad(args.adata_path)
    # filter control probes
    adata[:, ~adata.var.gene_symbol.str.contains("Blank")]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        metadata = pd.read_csv(args.metadata_path)

    metadata["cell_label"] = metadata["cell_label"].astype(str)
    metadata["x"] = metadata["x_reconstructed"] * 100
    metadata["y"] = metadata["y_reconstructed"] * 100

    metadata = metadata[
        [
            "cell_type",
            "cell_label",
            "x",
            "y",
            "brain_section_label",
        ]
    ]

    metadata["cell_type"] = LabelEncoder.fit_transform(metadata["subclass"])
    metadata["cell_type"] = metadata["cell_type"].astype(int)

    metadata = metadata.reset_index(drop=True)

    adata = adata[metadata["cell_label"]]
    patch_size = (17, 17)
    bs = 32

    sampler = CenterMaskSampler(
        metadata=metadata,
        adata=adata,
        patch_size=patch_size,
        cell_id_colname="cell_label",
        cell_type_colname="cell_type",
        tissue_section_colname="brain_section_label",
        max_num_cells=250,
    )

    loader = torch.utils.data.DataLoader(
        sampler,
        batch_size=bs,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        collate_fn=collate,
        prefetch_factor=4,
    )

    config_dct = OmegaConf.load(args.config_path)

    model = instantiate(config_dct.model)
    model = model.cuda()
    model.put_device = "cuda"

    checkpoint = torch.load(args.checkpoint_path)
    state_dict = {}
    model_state_dict = checkpoint['state_dict']
    for key in model_state_dict['state_dict']:
        key_wo_model = key.replace('model.', '')
        state_dict[key_wo_model] = model_state_dict[key]
    # if continuing after Lightning checkpoint need to change keys
    # otherwise can instantiate Lightning object and use that for 
    # embedding generation

    model.load_state_dict(state_dict, strict=True)

    # compiling model didn't seem to improve throughput for me; but ymmv
    # model = torch.compile(model)

    embeds = []

    with torch.inference_mode():
        for batch in tqdm.tqdm(loader):
            fwd = model(batch)
            embed = fwd["neighborhood_repr"].detach().cpu()
            embeds.append(embed)

    embeds = torch.cat(embeds)
    torch.save(embeds, "aibs-embeds.pth")

if __name__ == "__main__":
    main()
