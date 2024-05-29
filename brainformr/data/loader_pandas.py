from collections import namedtuple
from random import shuffle
from typing import List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from jaxtyping import Float, Int

NeighborhoodMetadata = namedtuple(
    "NeighborhoodMetadata",
    (
        "observed_expression",
        "masked_expression",
        "observed_cell_type",
        "masked_cell_type",
        "num_cells_obs",
    ),
)


def random_indices_from_series(df: pd.Series, n: int) -> List[int]:
    indices = list(range(len(df)))
    shuffle(indices)
    return indices[:n]


def index_outer_product(n: int) -> Int[np.ndarray, "seqlen 2"]:  # noqa: F722
    i = np.arange(n)
    grid = np.dstack(np.meshgrid(i, i)).reshape(-1, 2)
    return grid


def collate(batched_metadata: NeighborhoodMetadata):
    observed_expression = []
    masked_expression = []
    observed_cell_type = []
    masked_cell_type = []
    observed_neighboorhood_lens = []

    tot_num_obs_cells = sum([metadata.num_cells_obs for metadata in batched_metadata])
    bs = len(batched_metadata)

    attn_mask = torch.ones(
        tot_num_obs_cells + bs * 2, tot_num_obs_cells + bs * 2, dtype=torch.bool
    )

    decoder_mask = torch.ones(bs * 2, bs * 2, dtype=torch.bool)

    offset = 0

    for i, metadata in enumerate(batched_metadata):
        num_hidden_cells = metadata.num_cells_obs
        pooling_tok_idx = (2 * bs) - i
        decoding_tok_idx = bs - i

        indices: Int[np.ndarray, "total_seqlen x_y"] = (  # noqa: F722
            index_outer_product(num_hidden_cells) + offset
        )

        pooling_indices: Int[np.ndarray, "tot_seqlen"] = np.full(  # noqa: F821
            len(indices), -pooling_tok_idx
        )

        # if more than one decoding (ie not just single masked cell)
        # need to form the product indices of (observed_cells, n_decoding_queries)
        # and (cls_tokens, decoding_queries)
        decoding_indices: Int[np.ndarray, "tot_seqlen"] = np.full(  # noqa: F821
            len(indices), -decoding_tok_idx
        )

        x = indices[:, 0]
        y = indices[:, 1]

        attn_mask[x, y] = False
        attn_mask[pooling_indices, y] = False
        attn_mask[x, pooling_indices] = False
        attn_mask[-pooling_tok_idx, -pooling_tok_idx] = False

        attn_mask[decoding_indices, y] = False
        attn_mask[x, decoding_indices] = False
        attn_mask[decoding_indices, decoding_indices] = False

        attn_mask[pooling_indices, decoding_indices] = False
        attn_mask[decoding_indices, pooling_indices] = False

        decoder_mask[i, bs + i] = False
        decoder_mask[bs + i, i] = False

        offset += num_hidden_cells

        observed_expression.append(metadata.observed_expression)
        masked_expression.append(metadata.masked_expression)
        observed_cell_type.append(metadata.observed_cell_type)
        masked_cell_type.append(metadata.masked_cell_type)
        observed_neighboorhood_lens.append(metadata.num_cells_obs)

    observed_expression = torch.cat(observed_expression)
    masked_expression = torch.cat(masked_expression)
    observed_cell_type = torch.cat(observed_cell_type).long()
    masked_cell_type = torch.cat(masked_cell_type).long()

    num_cells_plus_cls = sum(observed_neighboorhood_lens) + bs

    return dict(
        observed_expression=observed_expression,
        masked_expression=masked_expression,
        observed_cell_type=observed_cell_type,
        masked_cell_type=masked_cell_type,
        observed_neighboorhood_lens=observed_neighboorhood_lens,
        full_mask=attn_mask,
        encoder_mask=attn_mask[:num_cells_plus_cls, :num_cells_plus_cls],
        decoder_mask=decoder_mask,
        bs=bs,
    )


class CenterMaskSampler(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata: Union[pd.DataFrame],
        adata: ad.AnnData,
        patch_size: Union[List[int], Tuple[int]],
        cell_id_colname: Optional[str] = "cell_label",
        cell_type_colname: Optional[str] = "cell_type",
        tissue_section_colname: Optional[str] = "brain_section_label",
        max_num_cells: Optional[Union[int, None]] = None,
    ):
        """_summary_

        Parameters
        ----------
        metadata : Union[pd.DataFrame]
            _description_
        adata : ad.AnnData
            _description_
        patch_size : Union[List[int], Tuple[int]]
            _description_
        cell_id_colname : Optional[str], optional
            _description_, by default "cell_label"
        cell_type_colname : Optional[str], optional
            _description_, by default "cell_type"
        tissue_section_colname : Optional[str], optional
            _description_, by default "brain_section_label"
        max_num_cells : Optional[Union[int, None]], optional
            _description_, by default None

        Raises
        ------
        TypeError
            _description_
        ValueError
            _description_
        TypeError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """
        if isinstance(adata, ad.AnnData) is False:
            raise TypeError(
                f"Input argument adata must be of type np.ndarray; got {type(adata)}."
            )

        if not all(
            (
                cell_type_colname in metadata.columns,
                cell_id_colname in metadata.columns,
                tissue_section_colname in metadata.columns,
            )
        ):
            raise ValueError(
                "Provided metadata has to have columns with columns specified by cell_id_colname, cell_type_colname and tissue_section_colname."
            )

        if isinstance(metadata, pd.DataFrame) is False:
            raise TypeError(
                f"Input argument metadata must be of type pl.DataFrame or pd.DataFrame; got {type(metadata)}."
            )

        if len(patch_size) != 2:
            raise ValueError(
                f"Input argument patch_size must be a tuple of length 2 with desired dims order (x, y, z); got {len(patch_size)}."
            )

        if len(metadata) != len(adata):
            raise ValueError("Metadata and adata must have the same length.")

        if not ("x" in metadata.columns and "y" in metadata.columns):
            raise ValueError(
                "Metadata must contain columns named x and y to index the spatial coordinates of the cells."
            )

        self.metadata = metadata

        self.cell_type_colname = cell_type_colname
        self.cell_id_colname = cell_id_colname
        self.tissue_section_colname = tissue_section_colname
        self.length = len(self.metadata)
        self.max_num_cells = np.inf if max_num_cells is None else max_num_cells

        self.adata = adata
        self.patch_size = np.array(patch_size).astype(int)

        self._preprocess_metadata()

    def __len__(self):
        return self.length

    def _preprocess_metadata(self):
        self.tissue_section_mapper = {
            section_label: subset_df
            for section_label, subset_df in self.metadata.groupby(
                self.tissue_section_colname
            )
        }

    def get_nearby_cells(
        self,
        centroid: npt.ArrayLike,
        patch_size: npt.ArrayLike,
        tissue_section: pd.DataFrame,
        discretize: bool = False,
    ) -> pd.DataFrame:
        start = centroid - (patch_size / 2)
        if discretize:
            start = np.rint(start).astype(int)

        end = start + patch_size

        lookup_x = tissue_section["x"].between(start[0], end[0])
        lookup_y = tissue_section["y"].between(start[1], end[1])

        return tissue_section[lookup_x & lookup_y]

    def idx_to_expr(self, indices: Union[str, int, List[Union[str, int]]]):
        return self.adata[indices].X

    def __getitem__(self, index) -> dict:
        metadata_row = self.metadata.iloc[index]
        index = metadata_row[self.cell_id_colname]

        centroid = np.array([metadata_row["x"], metadata_row["y"]])
        tissue_section: Union[pd.DataFrame, None] = self.tissue_section_mapper.get(
            metadata_row[self.tissue_section_colname], None
        )

        if tissue_section is None:
            raise RuntimeError(
                "Tissue section not found in `tissue_section_mapper` dict."
            )

        all_neighborhood_cells = self.get_nearby_cells(
            centroid, self.patch_size, tissue_section, discretize=True
        )

        if len(all_neighborhood_cells) == 1:
            masked_expression: Float[np.array, "one n_genes"] = self.idx_to_expr(index)  # noqa: F722
            masked_cell_type: int = metadata_row[self.cell_type_colname]

            neighborhood_metadata = NeighborhoodMetadata(
                observed_expression=torch.FloatTensor([]),
                masked_expression=torch.from_numpy(masked_expression),
                observed_cell_type=torch.LongTensor([]),
                masked_cell_type=torch.from_numpy([masked_cell_type]).long(),
                num_cells=1,
            )
            return neighborhood_metadata

        return self.neighborhood_gather(all_neighborhood_cells, index)

    def neighborhood_gather(
        self,
        neighborhood_cells: pd.DataFrame,
        ref_cell_label: str,
        as_dict: bool = False,
    ):
        neighborhood_cells = neighborhood_cells.reset_index(drop=True)
        neighborhood_cells_indices = neighborhood_cells[self.cell_id_colname]
        expression = self.adata[neighborhood_cells_indices].X

        num_cells_neighborhood = len(neighborhood_cells)

        observed_cells: pd.DataFrame = neighborhood_cells[
            neighborhood_cells_indices != ref_cell_label
        ]

        masked_cell = neighborhood_cells[neighborhood_cells_indices == ref_cell_label]

        if self.max_num_cells < num_cells_neighborhood - 1:
            random_indices: List[int] = random_indices_from_series(
                observed_cells[self.cell_id_colname], self.max_num_cells
            )
            observed_cells = observed_cells.iloc[random_indices]

        observed_cell_types = observed_cells[self.cell_type_colname].values
        observed_expression: npt.ArrayLike = expression[observed_cells.index]

        masked_cell_types = masked_cell[self.cell_type_colname].values
        masked_expression: npt.ArrayLike = expression[masked_cell.index]

        num_cells_obs = len(observed_cells)

        if as_dict:
            return dict(
                observed_expression=observed_expression,
                masked_expression=masked_expression,
                observed_cell_type=observed_cell_types,
                masked_cell_type=masked_cell_types,
                num_cells_obs=num_cells_obs,
            )
        else:
            return NeighborhoodMetadata(
                torch.from_numpy(observed_expression),
                torch.from_numpy(masked_expression),
                torch.from_numpy(observed_cell_types).long(),
                torch.from_numpy(masked_cell_types).long(),
                num_cells_obs,
            )
