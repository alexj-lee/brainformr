In order to train on the Zhuang lab data please see the config files labeled "Zhuang".

The setup is more or less the same as for the AIBS case. One easy solution is to simply load each metadata / `anndata` object indepently as a `brainformr.data.SimpleMaskSampler` object and concatenate them together post-hoc using `torch.utils.data.ConcatDataset`. This is what we do.

For reference, here is a snippet of code from the `ZhuangTrainer.load_data` function (see `scripts/training/train_zhuang.py`):

```
def load_data(self, config: DictConfig):

	all_dfs = []
	all_cls = set()

	for df_path in config.data.metadata_path:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", pd.errors.DtypeWarning)
			metadata = pd.read_csv(df_path)
			
		metadata["x"] = metadata["x"] * 100
		metadata["y"] = metadata["y"] * 100

		metadata = metadata.reset_index(drop=True)

		all_cls.update(metadata[config.data.celltype_colname].unique())
		all_dfs.append(metadata)

	le = LabelEncoder()
	le.fit(sorted(all_cls))

	trn_samplers = []
	valid_samplers = []

	for df, anndata_path in zip(all_dfs, config.data.adata_path):
		df["cell_type"] = le.transform(df[config.data.celltype_colname])
		df["cell_type"] = df["cell_type"].astype(int)

		df['cell_label'] = df['cell_label'].astype(str)

		df = df[['cell_type', 'cell_label', 'x', 'y', 'brain_section_label']]

		adata = ad.read_h5ad(anndata_path)
		adata = adata[df["cell_label"]]

		train_indices, valid_indices = train_test_split(
			range(len(adata)), train_size=config.data.train_pct
		)

		train_sampler = CenterMaskSampler(
			metadata=df,
			adata=adata,
			patch_size=config.data.patch_size,
			cell_id_colname=config.data.cell_id_colname,
			cell_type_colname="cell_type",
			tissue_section_colname=config.data.tissue_section_colname,
			max_num_cells=config.data.neighborhood_max_num_cells,
			indices=train_indices,
		)

		valid_sampler = CenterMaskSampler(
			metadata=df,
			adata=adata,
			patch_size=config.data.patch_size,
			cell_id_colname=config.data.cell_id_colname,
			cell_type_colname="cell_type",
			tissue_section_colname=config.data.tissue_section_colname,
			max_num_cells=config.data.neighborhood_max_num_cells,
			indices=valid_indices,
		)

		trn_samplers.append(train_sampler)
		valid_samplers.append(valid_sampler)

	train_loader = torch.utils.data.DataLoader(
		torch.utils.data.ConcatDataset(trn_samplers),
		batch_size=config.data.batch_size,
		num_workers=config.data.num_workers,
		pin_memory=False,
		shuffle=True,
		collate_fn=collate,
		prefetch_factor=4,  
	)
```

Otherwise, the overall process is the same. We load each metadata CSV file in the first portion of the code in order to collect its observed neuron types for conversion from string encoded labels to integer ones. If this is done on a per-metadata/dataframe basis, there will be inconsistency in the coding because the different mice don't all have all of the total cell types, in some cases.