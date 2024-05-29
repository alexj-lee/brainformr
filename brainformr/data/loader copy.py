from typing import List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch


class CenterMaskSampler(torch.utils.data.Dataset):
    def __init__(
        self,
        adata: ad.AnnData,
        patch_size: Union[List[int], Tuple[int]],
        cell_type_colname: Optional[str] = "cell_type",
        tissue_section_colname: Optional[str] = "brain_section_label"
    ):
        
        if cell_type_colname not in adata.obs:
            raise ValueError(
                f"Metadata (.obs) does not contain column {cell_type_colname}. This column is necessary to reference the scRNA-seq cell type labels."
            )
        
        if tissue_section_colname not in adata.obs:
            raise ValueError(
                f'Metadata (.obs) does not contain column {tissue_section_colname}. This column is necessary to index the different tissue sections.'
            )

        if len(patch_size) != 2:
            raise ValueError(
                f"Input argument patch_size must be a tuple of length 2 with desired dims order (x, y, z); got {len(patch_size)}."
            )
        
        if not ('x' in adata.obs.columns and 'y' in adata.obs.columns):
            raise ValueError('Metadata (.obs) must contain columns named x and y to index the spatial coordinates of the cells.')
        
        self.adata = adata
        self.cell_type_colname = cell_type_colname 
        self.length = len(self.metadata)
        
        self.patch_size = np.array(patch_size).astype(int)

        self._preprocess_metadata()

    def __len__(self):
        return self.length

    def _preprocess_metadata(self):
        self.tissue_section_mapper = {section_label: subset_df for section_label, subset_df in self.adata.obs.groupby(self.tissue_section_colname)}

    def get_nearby_cells(self, centroid: npt.ArrayLike, patch_size: npt.ArrayLike, tissue_section: pd.DataFrame, discretize: bool = False) -> pd.DataFrame:
        start = centroid - (patch_size / 2)
        if discretize:
            start = np.rint(start).astype(int)
        
        end = start + patch_size

        lookup_x = tissue_section.x.between(start[0], end[0])
        lookup_y = tissue_section.y.between(start[1], end[1])

        return tissue_section[lookup_x & lookup_y]
        
    def __getitem__(self, index) -> dict:

        metadata_row = self.adata.obs[index]
        index = metadata_row.index 

        # centroid = metadata_row[['x', 'y']].values
        tissue_section: Union[pd.DataFrame, None] = self.tissue_section_mapper.get(metadata_row[self.tissue_section_colname], None)
        
        if tissue_section is None:
            raise RuntimeError('Tissue section not found in `tissue_section_mapper` dict.')
        
        # nearby_cells = self.get_nearby_cells(centroid, self.patch_size, tissue_section, discretize=False).index
        









 
        





 
        


