import os
import sys
import pathlib
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
import anndata
import scanpy as sc

# import atlasapprox as aa
# from Bio import Phylo
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


species_full_dict = {
    "dmel": "Drosophila melanogaster",
    "znev": "Zootermopsis nevadensis",
    "ofor": "Odontotermes formosansus",
    "mdar": "Mastotermes darwiniensis",
    "hsjo": "Hodotermopsis sjostedti",
    "gfus": "Glyptotermes fuscus",
    "imin": "Incisitermes minor",
    "cfor": "Coptotermes formosanus",
    "nsug": "Neotermes sugioi",
    "pnit": "Pericapritermes nitobei",
    "cpun": "Cryptocercus punctulatus",
    "roki": "Reticulitermes okinawus",
    "rspe": "Reticulitermes speratus",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Save the result")
    args = parser.parse_args()

    print("Loading annotated h5ad with all termite species...")
    h5ad_fn = "data/saturn_output/final_adata_with_umap_and_dmel_rough_annotations.h5ad"
    adata = anndata.read_h5ad(h5ad_fn)

    print("Loading annotated h5ad with Znev cell types...")
    h5ad_znev_fn = "data/znev/combined_no_norm_clustered_new_znev.h5ad"
    adata_znev = anndata.read_h5ad(h5ad_znev_fn)

    print("Update znev annotations")
    adata.obs["labels_only_znev"] = adata.obs["labels_only_znev"].astype(str)
    adata.obs.loc[adata_znev.obs_names, "labels_only_znev"] = adata_znev.obs[
        "paper_cell_type_annotation"
    ].astype(str)
    adata.obs["labels_only_znev"] = adata.obs["labels_only_znev"].astype("category")

    print("Plotting UMAP colored by species...")
    fig, axs = plt.subplots(1, 3, figsize=(38, 10))
    sc.pl.umap(
        adata,
        color="species",
        show=False,
        size=20,
        ax=axs[0],
    )
    sc.pl.umap(
        adata,
        color="labels_only_znev",
        show=False,
        size=20,
        ax=axs[1],
    )
    sc.pl.umap(
        adata,
        color="leiden",
        show=False,
        size=20,
        ax=axs[2],
    )
    fig.tight_layout()

    print("Clustering looks ok in terms of resolution, now give names...")
    abundances = (
        adata.obs.groupby(["leiden", "labels_only_znev"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    del abundances["unknown (not znev)"]

    palette = dict(
        zip(
            adata.obs["labels_only_znev"].cat.categories,
            adata.uns["labels_only_znev_colors"],
        )
    )
    fig, axs = plt.subplots(6, 4, figsize=(24, 12))
    axs = axs.ravel()
    for ax, clu in zip(axs, abundances.index):
        tmp = abundances.loc[clu].nlargest(3)
        colors = [palette[i] for i in tmp.index]
        ax.barh(tmp.index.astype(str), tmp.values, color=colors)
        ax.set_title(f"Leiden {clu}")
        ax.invert_yaxis()
        if tmp.max() < 100:
            ax.set_xlim(0, 100)
    fig.tight_layout()

    znev_missing = 0
    cell_type_map = {}
    for clu in abundances.index:
        tmp = abundances.loc[clu].nlargest(1)
        if tmp.values[0] < 50:
            znev_missing += 1
            cell_type_map[clu] = f"znev_missing_{znev_missing}"
            continue
        tgt = tmp.index[0]
        if tgt.isdigit():
            tgt = f"znev_{tgt}"
        cell_type_map[clu] = tgt

    adata.obs["cell_type_znev_based"] = (
        adata.obs["leiden"].map(cell_type_map).astype("category")
    )

    print("Plotting UMAP colored by znev-based cell type...")
    fig, axs = plt.subplots(1, 3, figsize=(38, 10))
    sc.pl.umap(
        adata,
        color="species",
        show=False,
        size=20,
        ax=axs[0],
    )
    sc.pl.umap(
        adata,
        color="cell_type_znev_based",
        show=False,
        size=20,
        ax=axs[1],
    )
    sc.pl.umap(
        adata,
        color="leiden",
        show=False,
        size=20,
        ax=axs[2],
    )
    fig.tight_layout()

    if args.write:
        adata.write(
            "data/saturn_output/final_adata_with_umap_and_znev-based_annotations.h5ad"
        )

    plt.ion()
    plt.show()
