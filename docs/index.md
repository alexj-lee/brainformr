# Getting started

This is documentation for code written as part of the manuscript ["Data-driven fine-grained region discovery in the mouse brain with transformers"](https://www.biorxiv.org/content/10.1101/2024.05.05.592608v1). 

## Installation

* `pip install git+github.com:alexj-lee/brainformr.git` or clone and pip install; alternatively use the Dockerfile -- `PyTorch` and some other heavy libraries are required, which could take a couple minutes (<10).

## TLDR get started:

1. change data paths in `config/data` or create new data yaml file in that folder
2. specify model architecture in `config/model` yaml file
3. implement the `load_data` method for `BaseTrainer` in `scripts/training/lightning_model.py` (see `train_zhuang.py` and `train_aibs_mouse.py` for reference)
4. add wandb key if desired  to top level config in `config`
5. copy boilerplate for initiating training from `train_aibs_mouse.py` or `train_zhuang.py` (ie code in `main` that); make sure to specify correct config file in the `@hydra.main` decorator

## Core code components and usage

### Config management with hydra
The main interface to the training code we wrote is through `hydra` (https://hydra.cc/), which is a configuration framework that uses yaml files to orchestrate and organize complex workflows. Please see the hydra documentation for more information.

The pipeline controls the basic training operations through these yaml files and Pytorch Lightning.

For example, `scripts/config/model/base.yaml` controls the parameters of the transformer itself, for example:

```
_target_: brainformr.model.CellTransformer
encoder_embedding_dim: 384
decoder_embedding_dim: 384

encoder_num_heads: 8
decoder_num_heads: 8
attn_pool_heads: 8

encoder_depth: 4
decoder_depth: 4

cell_cardinality: 384
eps: 1e-9

n_genes: 500
xformer_dropout: 0.0
bias: True
zero_attn: True
```

& we can use hydra to directly instantiate this model by specifiying the object class, here `brainformr.model.CellTransformer`. What this looks like in context is in following snippet:

```
cfg_path = 'config.yaml'
cfg = OmegaConf.load(cfg_path)
model = hydra.utils.instantiate(cfg.model)

# model will have 500 gene output decoder depth of 4, etc. and will be an instance of class `CellTransformer`
```

### Composition of config files is controlled at top-level using another config

An example of this composition at high level is the `scripts/config/example.yaml` file, which contains the settings used to train on the Allen Institute for Brain Science MERFISH data in the [Allen Brain Cell Atlas](https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas). Note that "mouse1.yaml" refers to a file inside the `config/data` directory. Correspondingly there is a `config/model/base.yaml` file that is specified by the below config, which is found one-level-up (ie in the `config` directory). 

```
defaults:
  - _self_
  - data: mouse1.yaml
  - model: base.yaml
  - optimization: base.yaml

checkpoint_dir: 
wandb_project: 
model_checkpoint: 
wandb_code_dir: 
```

Where you can see we can group and order config components and define several high level attributes such as the checkpoint directory. You may like, however, to change these. For example including `wandb_project` will assume you can `wandb.login()` (see the wandb website for information on wandb and how to get a free account) and set this as the project. 

The `wandb_code_dir` argument will be used later to log the specific code used. 

For some files, a field may read: `???` indicating that field must be filled or `hydra` will error. 

Overall, fields in config files are accessible as `.[attribute]` in the DictConfig (from Omegaconf) object for example `config.model.n_genes`.

Keep in mind that for dataset paths, all of them ought to be hardcoded in. Therefore, for datapaths in `config/data` these paths should be considered placeholders for you to fill in. I left in paths to explicitly indicate filepaths to the Zhuang and AIBS MERFISH data hosted on [https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html).

### Training on the AIBS MERFISH data

The entrypoint to the training used for the Allen Institute for Brain Science MERFISH dataset (mouse 6388550) is in `scripts/train_aibs_mouse.py`, which uses `scripts/config/aibs1.yaml`. To run the code (assuming the package has been installed):

1. download the data (use `scripts/download_aibs.sh`)
2. edit `config/data/mouse1.yaml`, specifically:
```
adata_path: './abc_dataset/C57BL6J-6388550-log2.h5ad'
metadata_path: './abc_dataset/cell_metadata_with_cluster_annotation.csv'
```
3. change whatever combination of checkpoint and `wandb` settings in `scripts/config/aibs1.yaml`
4. run the trainer script (`scripts/training/train_aibs_mouse.py`)

```
chmod +x scripts/download_aibs.sh
./scripts/download_aibs.sh
python scripts/training/train_aibs_mouse.py
```

If this is useful to you, please consider citing our preprint:

```
@article {Lee2024.05.05.592608,
        author = {Alex Jihun Lee and Shenqin Yao and Nicholas Lusk and Hongkui Zeng and Bosiljka Tasic and Reza Abbasi-Asl},
        title = {Data-driven fine-grained region discovery in the mouse brain with transformers},
        elocation-id = {2024.05.05.592608},
        year = {2024},
        doi = {10.1101/2024.05.05.592608},
        publisher = {Cold Spring Harbor Laboratory},
        abstract = {Technologies such as spatial transcriptomics offer unique opportunities to define the spatial organization of the mouse brain. We developed an unsupervised training scheme and novel transformer-based deep learning architecture to detect spatial domains in mouse whole-brain spatial transcriptomics data. Our model learns local representations of molecular and cellular statistical patterns. These local representations can be clustered to identify spatial domains within the brain from coarse to fine-grained. Discovered domains are spatially regular, even with several hundreds of spatial clusters. They are also consistent with existing anatomical ontologies such as the Allen Mouse Brain Common Coordinate Framework version 3 (CCFv31) and can be visually interpreted at the cell type or transcript level. We demonstrate our method can be used to identify previously uncatalogued subregions, such as in the midbrain, where we uncover gradients of inhibitory neuron complexity and abundance. We apply our method to a separate multi-animal whole-brain spatial transcriptomic dataset and observe that inclusion of both sagittal and coronal tissue slices in region identification improves correspondence of spatial domains to CCF.Competing Interest StatementThe authors have declared no competing interest.},
        URL = {https://www.biorxiv.org/content/early/2024/05/07/2024.05.05.592608},
        eprint = {https://www.biorxiv.org/content/early/2024/05/07/2024.05.05.592608.full.pdf},
        journal = {bioRxiv}
}
```