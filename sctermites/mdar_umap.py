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
from adjustText import adjust_text


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
    print("Loading annotated h5ad with mdar...")
    h5ad_fn = "data/all_species/h5ads_processed/mdar.h5ad"
    adata = anndata.read_h5ad(h5ad_fn)

    print("Plotting UMAP colored by znev-based cell type...")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    sc.pl.umap(
        adata,
        color="cell_type_znev_based",
        show=False,
        size=20,
        ax=axs[0],
    )
    sc.pl.umap(
        adata,
        color="caste",
        show=False,
        size=20,
        ax=axs[1],
    )
    fig.tight_layout()

    plt.ion()
    plt.show()

    adata.obs["cell_type"] = adata.obs["cell_type_znev_based"].cat.rename_categories(
        {"muscle cell": "muscle"}
    )
    fig, ax = plt.subplots(figsize=(3.4, 3))
    sc.pl.umap(
        adata,
        color="cell_type",
        groups=["muscle", "neuron", "T1"],
        show=False,
        size=30,
        ax=ax,
        legend_loc="upper left",
        add_outline=True,
        palette=["tomato", "steelblue", "deeppink"],
    )
    ax.legend(frameon=False)
    ax.get_legend().get_texts()[-1].set_text("other")
    ax.set_title("")
    ax.set_axis_off()
    fig.tight_layout()

    fig.savefig("figures/umap_mdar.svg")
    fig.savefig("figures/umap_mdar.png")
