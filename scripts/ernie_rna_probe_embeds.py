import argparse

import paddle
from rna_ernie import BatchConverter
from paddlenlp.transformers import ErnieModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_file", type=str, help="Input file containing RNA probe names in string format.", 
                            required=True)
    parser.add_argument('--')
    return parser.parse_args()