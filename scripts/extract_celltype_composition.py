import argparse

import pandas as pd
import numpy as np
from sklearn import preprocessing
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--metadata_path", type=str, help="Path to metadata file. We expect a CSV.", required=True)
    parser.add_argument('--celltype_level', type=str, help='The level of cell type to extract.', required=True)

    workflow_type = parser.add_mutually_exclusive_group(required=True)

    workflow_type.add_argument('--labels_path', type=str, help='Path to labels file. We expect an `npy` file.', required=True)
    workflow_type.add_argument('--colname', type=str, help='Column name in metadata file that contains the labels.', required=True)

    parser.add_argument('--output_fname', type=str, help='Filename to save the plot.', required=True)

    return parser.parse_args()

def main():
    args = parse_args()

    try:
        metadata = pd.read_csv(args.metadata_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    
    if args.celltype_level not in metadata.columns:
        raise ValueError(f"Column name not found in metadata: {args.celltype_level}")
    
    if args.labels_path:
        try:
            labels = np.load(args.labels_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Labels file not found: {args.labels_path}")
        
        metadata['region'] = preprocessing.LabelEncoder().fit_transform(labels) # these will be in order anyway, but just in case

    elif args.colname:
        if args.colname not in metadata.columns:
            raise ValueError(f"Column name not found in metadata: {args.colname}")
        
        metadata['region'] = preprocessing.LabelEncoder().fit_transform(metadata[args.colname])
        
    # we need to fit the encoders so that it's easy to then later index into the `celltype_composition_matrix`
    # otherwise I suppose we could have just kept two dictionaries or something 

    l_enc_celltype = preprocessing.LabelEncoder()
    l_enc_celltype.fit(sorted(metadata[args.celltype_level].unique()))

    metadata['celltype_tocount'] = l_enc_celltype.transform(metadata[args.celltype_level])

    num_rows = metadata['region'].nunique()
    num_cols = metadata['celltype_tocount'].nunique()

    celltype_composition_matrix = np.zeros((num_rows, num_cols), dtype=np.uint32)
    pbar = tqdm.tqdm(metadata.groupby('brain_section_label'))
    for region_index, groupby in pbar:
        pbar.set_description(f"Processing {region_index}")
        value_counts = groupby['celltype_tocount'].value_counts()
        celltype_composition_matrix[region_index, list(value_counts.keys())] = list(value_counts.values())
    
    np.save(args.output_fname, celltype_composition_matrix)

if __name__ == '__main__':
    main()