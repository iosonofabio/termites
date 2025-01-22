"""Prepare the h5ads for SATURN.

This involved merging the samples of distinct castes for each species (if necessary), then clustering the cells.

SATURN requires this for the weak supervision in which *all* species must have an annotation. It's the coarse grained equivalent of SAMap's neighborhoods.

This script requires anndata and scanpy. One way to do that is to use the SATURN conda environment:

source ~/miniconda3/bin/activate && conda activate saturn
"""

import os
import pathlib
import argparse
import numpy as np
import pandas as pd
import glob
import anndata
import scanpy as sc
import matplotlib.pyplot as plt


# NOTE: this is the table with the closest known genomes as of the time of writing. In the "docs" folder there is
# genomes_termite_data_usb_scRNA.ods
sample_dict = {
    # Perfect genome matches
    "dmel": ["dmel.h5ad", "Dmel_gene_all_esm1b.pt"],
    "znev": ["znev.h5ad", "Znev_gene_all_esm1b.pt"],
    "ofor": ["ofor.h5ad", "Ofor_gene_all_esm1b.pt"],
    "mdar": ["mdar.h5ad", "Mdar_gene_all_esm1b.pt"],
    "hsjo": ["hsjo.h5ad", "Hsjo_gene_all_esm1b.pt"],
    "gfus": ["gfus.h5ad", "Gfus_gene_all_esm1b.pt"],
    # Imperfect genome matches
    "imin": ["imin.h5ad", "Isch_gene_all_esm1b.pt"],
    "cfor": ["cfor.h5ad", "Cges_gene_all_esm1b.pt"],
    "nsug": ["nsug.h5ad", "Ncas_gene_all_esm1b.pt"],
    "pnit": ["pnit.h5ad", "Punk_gene_all_esm1b.pt"],
    "cpun": ["cpun.h5ad", "Cmer_gene_all_esm1b.pt"],
    "roki": ["roki.h5ad", "Rfla_gene_all_esm1b.pt"],
    "rspe": ["rspe.h5ad", "Rfla_gene_all_esm1b.pt"],
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    args = parser.parse_args()

    h5ad_input_fdn = pathlib.Path(
        "/mnt/data/projects/termites/data/sc_termite_data/h5ads"
    )
    output_fdn = pathlib.Path(
        "/mnt/data/projects/termites/data/sc_termite_data/saturn_data/h5ad_by_species"
    )

    os.makedirs(output_fdn, exist_ok=True)

    for species, datum in sample_dict.items():
        print(species)
        h5ad_fn = output_fdn / datum[0]
        if (not args.overwrite) and h5ad_fn.exists():
            print("File exists. Skipping.")
            continue

        # Merge castes
        adatas = []
        fns = glob.glob(str(h5ad_input_fdn) + f"/{species}*.h5ad")
        if len(fns) == 0:
            print(f"  WARNING! No files found for {species}")
            continue

        for fn in fns:
            caste = pathlib.Path(fn).stem.split("_")[1]
            if caste == "roach":
                caste = "none"
            print("  " + caste)
            adata = anndata.read_h5ad(fn)
            adata.obs_names_make_unique()
            if caste != "combined":
                adata.obs["caste"] = caste
            else:
                # This is znev
                adata.obs["cell_type"] = adata.obs["samap_annot"]
            adatas.append(adata)
        if len(adatas) == 1:
            adata = adatas[0]
            adata.obs["sample"] = 0
            adata.obs["sample"] = pd.Categorical(adata.obs["sample"])
        else:
            adata = anndata.concat(adatas, label="sample")
            adata.obs_names_make_unique()
            adata.var = adatas[0].var
            print(adata.obs)

        # FIXME: somehow Znev has only embeddings for 30% of the genes, around 5700
        if species == "znev":
            embeddings_summary_fdn = pathlib.Path(
                "/mnt/data/projects/termites/data/sc_termite_data/saturn_data/esm_embeddings_summaries/"
            )
            embedding_fn = sample_dict["znev"][1]
            embedding_fn = embeddings_summary_fdn / embedding_fn
            embedding = __import__("torch").load(embedding_fn)
            features = pd.Index(embedding.keys())
            adata = adata[:, features]

        if "cell_type" in adata.obs.columns:
            print("already annotated, writing output directly")
            adata.write(h5ad_fn)
            continue

        print("Preprocessing")
        print("  Filter cells/genes")
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=3)

        print("  Store raw counts")
        adata.layers["counts"] = adata.X.copy()

        print("  Normalise cptt")
        sc.pp.normalize_total(adata)

        print("  Log1p")
        sc.pp.log1p(adata)

        print("  HVG")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")

        print("  PCA")
        sc.tl.pca(adata)

        print("  KNN")
        sc.pp.neighbors(adata)

        print("  UMAP")
        sc.tl.umap(adata)

        print("Leiden clustering")
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

        print("Plot umap colored by cluster")
        plt.ion()
        plt.close("all")
        sc.pl.umap(adata, color=["leiden", "caste"])
        fig = plt.gcf()
        axs = fig.axes
        for ax in axs:
            ax.set_axis_off()
        fig.suptitle(species)
        fig.tight_layout()

        # Confirm the clustering manually
        confirm = None
        while confirm not in ("", "y", "n", "b"):
            if confirm is not None:
                print("Invalid input.")
            confirm = input("Confirm? ([y]es/[n]o/b[reak]): (y)")[:1]
        if confirm == "b":
            print("Breaking.")
            break

        confirm = not (confirm == "n")
        plt.close(fig)

        if not confirm:
            print("Clustering not confirmed. Skipping.")
            continue

        # Assign cell type
        adata.obs["cell_type"] = adata.obs["leiden"]

        # Restore original counts
        adata.X = adata.layers["counts"]

        # Write output to file
        adata.write(h5ad_fn)
