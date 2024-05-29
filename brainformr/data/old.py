from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


def get_nearby_cells(centroid: npt.ArrayLike, patch_size: npt.ArrayLike, tissue_section: pd.DataFrame, discretize: bool = False) -> pd.DataFrame:
    start = centroid - (patch_size / 2)
    if discretize:
        start = np.rint(start).astype(int)
    
    end = start + patch_size

    lookup_x = tissue_section.x.between(start[0], end[0])
    lookup_y = tissue_section.y.between(start[1], end[1])

    return tissue_section[lookup_x & lookup_y]
        
class CenterMaskSampler(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        gex_matrix: ad.AnnData,
        patch_size: Union[List[int], Tuple[int]],
        cell_type_colname: Optional[str] = "subclass",
        tissue_section_colname: Optional[str] = "brain_section_label"
    ):
        if isinstance(gex_matrix, np.ndarray) is False:
            raise TypeError(
                f"Input argument gex_matrix must be of type np.ndarray; got {type(gex_matrix)}."
            )

        if cell_type_colname not in ("cluster", "supertype", "subclass", "class"):
            
            raise ValueError(
                f'Input argument cell_type_colname must be one of "cluster", "supertype", "subclass", or "class"; got {cell_type_colname}.'
                
            )
        cell_type_colname = cell_type_colname.lower()
        
        if cell_type_colname not in metadata.columns:
            
            raise ValueError(
                f"Metadata does not contain column {cell_type_colname}. This column is necessary to reference the scRNA-seq cell type labels."
                
            )
        
        if tissue_section_colname not in metadata.columns:
            raise ValueError(
                f'Metadata does not contain column {tissue_section_colname}. This column is necessary to index the different tissue sections.'
            )

        if isinstance(metadata, pd.DataFrame) is False:
            raise TypeError(
                f"Input argument metadata must be of type pd.DataFrame; got {type(metadata)}."
            )
        
        if len(patch_size) != 2:
            raise ValueError(
                f"Input argument patch_size must be a tuple of length 2 with desired dims order (x, y, z); got {len(patch_size)}."
            )
        
        if len(metadata) != len(gex_matrix):
            raise ValueError('Metadata and gex_matrix must have the same length.')
        
        if not ('x' in metadata.columns and 'y' in metadata.columns):
            raise ValueError('Metadata must contain columns named x and y to index the spatial coordinates of the cells.')
        
        self.metadata = metadata
        self.cell_type_colname = cell_type_colname 
        self.length = len(self.metadata)
        
        self.X = gex_matrix
        self.patch_size = np.array(patch_size).astype(int)

        self._preprocess_metadata()

    def __len__(self):
        return self.length

    def _preprocess_metadata(self):
        self.cell_type_lencoder = LabelEncoder()
        self.metadata['cell_type_ohe'] = self.cell_type_lencoder.fit_transform(self.metadata[self.cell_type_colname])

        self.tissue_section_mapper = {section_label: subset_df for section_label, subset_df in self.metadata.groupby(self.tissue_section_colname)}

    def __getitem__(self, index) -> dict:

        metadata_row = self.metadata.iloc[index]
        centroid = metadata_row[['x', 'y']].values
        tissue_section: Union[pd.DataFrame, None] = self.tissue_section_mapper.get(metadata_row[self.tissue_section_colname], None)
        
        if tissue_section is None:
            raise RuntimeError('Tissue section not found in `tissue_section_mapper` dict.')
        
        nearby_cells = get_nearby_cells(centroid, self.patch_size, tissue_section, discretize=False)
        nearby_cells_indices = nearby_cells[self.cell_type_colname]








 
        





 
        


