## Dataloader information
We use a packed sequence format, so the returns from this function are not in general `(batch_size, length, features)` tensor but a `(length, features)` matrix. 

These are the arguments for the loader we implement:

::: brainformr.data.CenterMaskSampler.__init__
	handler: python
	options:
	  show_source: false
	  show_root_heading: false
	  merge_init_into_class: true
	  

The required data components are an `anndata` object with keys corresponding to columns in a dataframe of metadata, `metadata`. 

The dataframe **must** have:

* x: spatial coordinates
* y: spatial coordinates
* one column (`cell_id_colname`) which refers to the keys, these will be used directly to subset into the `anndata` object
* one column (`cell_type_colname`) which is a class encoded integer corresponding to the single cell classes from the non-spatial scRNA-seq data clustering 

The units of `patch_size` are not scaled internally, so they must "match" the units of `x` and `y`. 

<br>

## Outputs of `__getitem__` and `collate`

These are the steps we implement so far:

1. extract spatial neighbors (within some radius) for the cell of interest
2. extract expression (from the `anndata` object) and the class encoding (`cell_type_colname`) for each cell 
3. partition them into two sets, which are simply lists in a `namedtuple`, see `brainformr.data.NeighborhoodMetadata`

Note that we then need to stack these together across batches and create attention matrices, which is done in `brainformr.data.loader_pandas.collate`. We create three attention matrices (we use a binary adjacency matrix):

| Syntax      | Description |
| ----------- | ----------- |
| `encoder_mask`   | allow the neighborhoods to only attend to each other       |
| `pooling_mask`   | pool into a single query token        |
| `decoder_mask`   | allow query token and decoding cell tokens to attend to each other				| 

See the function documentation and the next section [model](model.md) for more information. 

My first-shot implementation of this workflow (in `pandas`) was ~1.3X faster than my next attempt, in `polars`. I welcome any feedback on why my `polars` code may have been suboptimal.
