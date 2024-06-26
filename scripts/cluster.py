# NOTE: you need cuml for this to work, which is not installed by default in this requirements.txt
# write me a script that will argparse one item (embeds_path) and load it using torch.load
# then use cuml to KMeans the embeddings into 10 clusters

import argparse

import cuml
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeds_path", type=str, help="Path to embeddings file.")
    parser.add_argument("--n_clust", type=int, help="Number of clusters")
    return parser.parse_args()


def main():
    args = parse_args()
    embeds = torch.load(args.embeds_path)
    embeds = embeds.numpy()

    kmeans = cuml.KMeans(n_clusters=args.n_clust)
    kmeans.fit(embeds)
    labels = kmeans.predict(embeds)

    np.save(f"./kmeans-cluster_labels-k{args.n_clust}.npy", labels)


if __name__ == "__main__":
    main()
