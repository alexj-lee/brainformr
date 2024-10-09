import argparse
import os
import warnings
import pathlib

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
    parser.add_argument("--adata_path", type=str, help="Path to anndata file.", required=True)
    parser.add_argument("--metadata_path", type=str, help="Path to metadata file.", required=True)
    parser.add_argument("--config_path", type=str, help="Path to config file.", required=True)
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint file from training.', required=True)
    parser.add_argument('--output_path', type=str, help='Path to save embeddings.', required=True)
    parser.add_argument('--celltype_colname', type=str, help='The column name that contains the cell type assignments', required=False, default='')
    parser.add_argument('--preprocess_type', type=str, help='The type of preprocessing to apply to the metadata', required=False, default='aibs')
    return parser.parse_args()

def preprocess_langlieb(adata, metadata):
    metadata.rename(columns={'Raw_Slideseq_X': 'x',
                            'Raw_Slideseq_Y': 'y',
                            'PuckID': 'brain_section_label',
                            }, inplace=True)

    metadata['cell_type'] = metadata['cell_index'].astype(int)
    metadata = metadata.reset_index(drop=True)

    metadata = metadata[["cell_type", 'cell_label', 'x', 'y', 'brain_section_label']]

    return adata, metadata

def preprocess_aibs(adata, metadata):

    cell_type_colname = 'subclass_name' # args.celltype_colname

    metadata["cell_label"] = metadata["cell_label"].astype(str)
    metadata['cell_type'] = metadata[cell_type_colname].astype(str)
    # metadata["x"] = metadata["x_reconstructed"] * 100
    # metadata["y"] = metadata["y_reconstructed"] * 100]
    metadata["x"] = metadata["spatial_x"] / 10.
    metadata["y"] = metadata["spatial_y"] / 10.

    metadata = metadata[
        [
            "cell_type",
            "cell_label",
            "x",
            "y",
            "brain_section_label",
        ]
    ]

    metadata = metadata.sort_values(by='brain_section_label')

    metadata["cell_type"] = LabelEncoder().fit_transform(metadata['cell_type'])
    metadata["cell_type"] = metadata["cell_type"].astype(int)

    metadata = metadata.reset_index(drop=True)


    return adata, metadata

def main():
    args = parse_args()
    if not os.path.exists(args.adata_path):
        raise FileNotFoundError(f"Anndata file not found: {args.adata_path}")

    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    
    adata = ad.read_h5ad(args.adata_path)
    # filter control probes
    adata[:, ~adata.var.index.str.contains("Blank")]
    #adata[:, ~adata.var.gene_symbol.str.contains("Blank")]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        metadata = pd.read_csv(args.metadata_path)

    if args.preprocess_type == 'aibs':
        adata, metadata = preprocess_aibs(adata, metadata)
    elif args.preprocess_type == 'langlieb':
        adata, metadata = preprocess_langlieb(adata, metadata)
    else:
        raise NotImplementedError(f"Preprocessing type {args.preprocess_type} not implemented.\n\
                                  Please pass one of ['aibs', 'langlieb']")

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        max_num_cells=None,
    )

    loader = torch.utils.data.DataLoader(
        sampler,
        batch_size=bs,
        shuffle=False,
        num_workers=1,
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
    for key in model_state_dict:
        if args.preprocess_type == 'aibs':
            key_wo_model = key.replace('model.', '')
        elif args.preprocess_type == 'langlieb':
            key_wo_model = key.replace('model', '').replace('_orig_mod.', '').lstrip('.')
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
    new_name = output_path/f'{pathlib.Path(args.adata_path).stem}_embeds_ps{patch_size[0]}.pth'
    torch.save(embeds, new_name)

if __name__ == "__main__":
    main()
