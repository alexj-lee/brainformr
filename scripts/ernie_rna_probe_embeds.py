import argparse
import pathlib

import paddle
from rna_ernie import BatchConverter
from paddlenlp.transformers import ErnieModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_fasta", type=str, help="Input file containing RNA probe names in string format.", 
                        )
    parser.add_argument('--output_file', type=str, help='Output file to save the embeddings in numpy format.')
    parser.add_argument('--vocab_path', type=str, help='Path to the vocabulary file.')
    parser.add_argument('--model_dir', type=str, help='Path to the model directory.')

    parser.add_argument('--bs', type=int, default=256, help='Batch size for inference.')
    parser.add_argument('--max_seqlen', type=int, default=512, help='Maximum sequence length for the model.')
    return parser.parse_args()

def exists(path):
    return pathlib.Path(path).exists()

def main():
    args = parse_args()

    for f in (args.embeddings_file, args.vocab_path, args.model_dir):
        if not exists(f):
            raise FileNotFoundError(f"{f} does not exist.")
    
    batch_converter = BatchConverter(vocab_path=args.vocab_path, 
                                     k_mer=1,
                                     batch_size=args.bs,
                                     max_seq_len=args.max_seqlen)
    
    model = ErnieModel.from_pretrained(args.model_dir)

    
    

if __name__ == "__main__":
    main()