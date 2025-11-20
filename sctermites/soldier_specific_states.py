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

protein_seqs_dict = {
    # Perfect genome matches
    "dmel": ["d_melanogaster.h5ad", "d_melanogaster_gene_all_esm1b.pt"],
    "znev": ["znev.h5ad", "Znev_proteins_rep.faa"],
    "ofor": ["ofor.h5ad", "Ofor_proteins_rep.faa"],
    "mdar": ["mdar.h5ad", "Mdar_proteins_rep.faa"],
    "hsjo": ["hsjo.h5ad", "Hsjo_proteins_rep.faa"],
    "gfus": ["gfus.h5ad", "Gfus_proteins_rep.faa"],
    # Imperfect genome matches
    "imin": ["imin.h5ad", "Isch_proteins_rep.faa"],
    "cfor": ["cfor.h5ad", "Cges_proteins_rep.faa"],
    "nsug": ["nsug.h5ad", "Ncas_proteins_rep.faa"],
    "pnit": ["pnit.h5ad", "Punk_proteins_rep.faa"],
    "cpun": ["cpun.h5ad", "Cmer_proteins_rep.faa"],
    "roki": ["roki.h5ad", "Rfla_proteins_rep.faa"],
    "rspe": ["rspe.h5ad", "Rfla_proteins_rep.faa"],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--celltype", default="muscle cell", help="Cell type to focus on"
    )
    parser.add_argument("--savefig", action="store_true", help="Save the figures")
    args = parser.parse_args()

    focus_cell_type = args.celltype

    print("Loading annotated h5ad with all termite species...")
    h5ad_fn = "data/saturn_output/final_adata_with_umap_and_znev-based_annotations_no_wrong_meta.h5ad"
    adata = anndata.read_h5ad(h5ad_fn)

    print(f"Focus on {focus_cell_type}...")
    adata_focus = adata[adata.obs["cell_type_znev_based"] == focus_cell_type]

    if False:
        print("Plotting UMAP colored by znev-based cell type...")
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        sc.pl.umap(
            adata_focus,
            color="species",
            show=False,
            size=20,
            ax=axs[0],
        )
        sc.pl.umap(
            adata_focus,
            color="caste",
            show=False,
            size=20,
            ax=axs[1],
        )
        sc.pl.umap(
            adata_focus[adata_focus.obs["species"] == "znev"],
            color="caste",
            show=False,
            size=30,
            add_outline=True,
            ax=axs[2],
        )
        fig.tight_layout()

        print("Look in greater detail at Znev across soldiers/nonsoldiers")
        adata_focus_znev = adata_focus[adata_focus.obs["species"] == "znev"]
        sc.tl.embedding_density(adata_focus_znev, basis="umap", groupby="caste")
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.ravel()
        for ax, caste in zip(axs, ["soldier", "worker", "king", "queen"]):
            sc.pl.embedding_density(
                adata_focus_znev,
                key="umap_density_caste",
                group=caste,
                show=False,
                # size=30,
                # add_outline=True,
                # ax=ax,
            )
        fig.tight_layout()

        print(f"Perform pseudotime on {focus_cell_type}...")
        xy_rootd = {
            "muscle cell": np.array([-5.842, -1.916]),
        }
        xy_root = xy_rootd[focus_cell_type]
        iroot = ((adata_focus.obsm["X_umap"] - xy_root) ** 2).sum(axis=1).argmin()
        adata_focus.uns["iroot"] = iroot
        sc.tl.diffmap(adata_focus)
        sc.tl.dpt(adata_focus)

        fig, ax = plt.subplots(figsize=(5, 5))
        sc.pl.umap(
            adata_focus,
            color="dpt_pseudotime",
            show=False,
            size=20,
            ax=ax,
        )
        fig.tight_layout()

        print("Plot distributions of species and caste on pseudotime")
        from scipy.stats import gaussian_kde

        palette = dict(
            zip(
                adata_focus.obs["caste"].cat.categories,
                adata_focus.uns["caste_colors"],
            )
        )
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(projection="3d")

        coffset = {"worker": -0.2, "soldier": 0.0, "king": 0.2, "queen": 0.4, "n/a": 0}
        data = adata_focus.obs.groupby("species", observed=True)
        species_focus = adata_focus.obs["species"].value_counts().index
        for i, species in enumerate(species_focus[::-1]):
            gby = data.get_group(species).groupby("caste", observed=True)
            for caste, datum in gby:
                color = palette[caste]
                offset = coffset[caste]
                ptime = datum["dpt_pseudotime"].values
                xmod = np.linspace(0, 1, 100)
                ymod = gaussian_kde(ptime, bw_method=0.2)(xmod)
                ymod = ymod / ymod.max() * 0.8  # normalize
                ax.fill_between(
                    xmod,
                    i + offset,
                    ymod,
                    xmod,
                    i + offset,
                    0,
                    color=color,
                    alpha=0.5,
                    label=f"{species}-{caste}",
                )
        ax.set_yticks(np.arange(len(species_focus)))
        ax.set_yticklabels(species_focus[::-1])
        fig.tight_layout()

    print(f"Check out {focus_cell_type} by caste, speces by species")
    ncells_by_species_and_caste = (
        adata.obs.groupby(["species", "caste", "cell_type_znev_based"], observed=True)
        .size()
        .unstack(fill_value=0)[focus_cell_type]
        .unstack(fill_value=0)
    )
    species_multicaste = ncells_by_species_and_caste.index[
        ncells_by_species_and_caste.gt(0).sum(axis=1) > 1
    ]

    print("Load and preprocess individual species")
    caste_dict = {"king": "king", "sol": "soldier", "roach": "n/a"}
    adatad = {}
    for species in species_multicaste:
        print(f"Loading {species}...")
        if species == "znev":
            print("  Znev already processed, load it and adjust...")
            adata_species = anndata.read_h5ad(
                "data/znev/combined_no_norm_clustered_new_znev.h5ad"
            )
            adata_species = adata_species[adata_species.obs_names.isin(adata.obs_names)]
            adata_species.obs["cell_type_znev_based"] = adata_species.obs[
                "paper_cell_type_annotation"
            ]
        else:
            h5ad_preprocessed = f"data/all_species/h5ads_processed/{species}.h5ad"
            if pathlib.Path(h5ad_preprocessed).exists():
                print("  Preprocessed file exists, load it...")
                adata_species = anndata.read_h5ad(h5ad_preprocessed)

            else:
                h5ad_fns = list(
                    pathlib.Path("data/all_species/h5ads").glob(f"{species}*.h5ad")
                )
                adatas_species = []
                for h5ad_fn in h5ad_fns:
                    caste = h5ad_fn.stem.split("_")[1].split(".")[0]
                    tmp = anndata.read_h5ad(h5ad_fn)
                    tmp.obs["species"] = species
                    tmp.obs["caste"] = caste_dict[caste]
                    # Restrict to cells that we have annotations for
                    tmp = tmp[tmp.obs_names.isin(adata.obs_names)]
                    tmp.obs["cell_type_znev_based"] = adata.obs.loc[
                        tmp.obs_names, "cell_type_znev_based"
                    ]
                    adatas_species.append(tmp)
                adata_species = anndata.concat(adatas_species)
                del adatas_species

                print(f"Preprocess {species}...")
                adata_species.obs["nUMI"] = adata_species.X.sum(axis=1)
                adata_species.obs["ngenes"] = (adata_species.X > 0).sum(axis=1)
                adata_species.layers["raw"] = adata_species.X.copy()

                print("  Get highly variable genes...")
                sc.pp.highly_variable_genes(
                    adata_species,
                    n_top_genes=1000,
                    flavor="seurat_v3",
                    subset=True,
                )

                print("  Normalize...")
                sc.pp.normalize_total(adata_species, target_sum=1e4)

                print("  Log...")
                sc.pp.log1p(adata_species)

                print("  PCA...")
                sc.tl.pca(adata_species, n_comps=50, svd_solver="arpack")

                print("  Neighbors...")
                sc.pp.neighbors(adata_species, n_neighbors=15, n_pcs=30)

                print("  UMAP...")
                sc.tl.umap(adata_species)

                adata_species.layers["normalized"] = adata_species.X.copy()

            adata_species.X = adata_species.layers["raw"]
            adata_species.write(h5ad_preprocessed)

        adata_species.X = adata_species.layers["normalized"]
        adatad[species] = adata_species

        if False:
            print("Visualize UMAP...")
            fig, ax = plt.subplots(figsize=(9, 5))
            sc.pl.umap(
                adata_species,
                color="cell_type_znev_based",
                ax=ax,
            )
            ax.set_title(species)
            fig.tight_layout()

    # NOTE: it's always king and soldier except for znev which has all 4 castes

    palette = {
        "king": "#1f77b4",
        "queen": "#ff7f0e",
        "soldier": "#2ca05c",
        "worker": "#962758",
        "n/a": "#7f7f7f",
    }
    fig, axs = plt.subplots(2, 3, figsize=(11, 8))
    axs = axs.ravel()
    for species, ax in zip(species_multicaste, axs):
        adata_species = adatad[species]
        xmin, ymin = adata_species.obsm["X_umap"].min(axis=0)
        xmax, ymax = adata_species.obsm["X_umap"].max(axis=0)
        adata_species.uns["caste_colors"] = [
            palette[caste] for caste in adata_species.obs["caste"].cat.categories
        ]
        adata_species_nonfocus = adata_species[
            adata_species.obs["cell_type_znev_based"] != focus_cell_type,
        ]
        adata_species_focus = adata_species[
            adata_species.obs["cell_type_znev_based"] == focus_cell_type,
        ]
        sc.pl.umap(
            adata_species_nonfocus,
            color=None,
            na_color=palette["n/a"],
            na_in_legend=False,
            size=40,
            alpha=0.2,
            ax=ax,
            show=False,
            zorder=0,
        )
        sc.pl.umap(
            adata_species_focus,
            color="caste",
            size=50,
            alpha=0.6,
            add_outline=True,
            ax=ax,
            show=False,
            zorder=1,
            legend_loc="best",
        )
        ax.set(
            xlim=(xmin - 0.5, xmax + 0.5),
            ylim=(ymin - 0.5, ymax + 0.5),
            xticks=[],
            yticks=[],
            xlabel="",
            ylabel="",
        )
        ax.set_title(species)
    fig.suptitle(f"{focus_cell_type} across species and castes", fontsize=16)
    fig.tight_layout()

    print("Differential expression between soldier and king...")
    degsd = {}
    for species_code, adata_species in adatad.items():
        adata_species_focus = adata_species[
            (adata_species.obs["cell_type_znev_based"] == focus_cell_type)
            & adata_species.obs["caste"].isin(["soldier", "king"]),
        ]
        sc.tl.rank_genes_groups(
            adata_species_focus,
            "caste",
            method="wilcoxon",
            groups=["soldier"],
            reference="king",
        )
        tmpi = adata_species_focus.uns["rank_genes_groups"]
        degs = pd.DataFrame(
            {
                "score": tmpi["scores"]["soldier"],
                "logfoldchanges": tmpi["logfoldchanges"]["soldier"],
                "pvals_adj": tmpi["pvals_adj"]["soldier"],
            },
            index=tmpi["names"]["soldier"],
        )
        degsd[species_code] = degs

    def get_protein_sequences(species, genes):
        """Extract protein sequence from fasta file for a species and gene."""
        from Bio.SeqIO import FastaIO

        genes_left = set(genes)
        seqs = {}

        fn = protein_seqs_dict[species][1]
        fn = f"data/all_species/protein_coding_genes_fasta/{fn}"
        with open(fn) as handle:
            for protein_name, seq in FastaIO.SimpleFastaParser(handle):
                gene_name = protein_name.split("-")[0]
                if gene_name in genes_left:
                    genes_left.remove(gene_name)
                    seqs[gene_name] = seq
                if len(genes_left) == 0:
                    break

        if len(seqs) == 0:
            raise ValueError(
                f"No sequences found for genes {genes} in species {species}"
            )
        return seqs

    def blast_genes_dmel(protein_seqd):
        import subprocess as sp
        from Bio import Blast

        gene_names, protein_seqs = zip(*protein_seqd.items())

        tmp_file = "/tmp/blasttmp/gene.fasta"
        tmp_fileout = "/tmp/blasttmp/blast_result.xml"

        with open(tmp_file, "w") as f:
            for gene_name, protein_seq in zip(gene_names, protein_seqs):
                f.write(f">{gene_name}\n{protein_seq}\n")

        sp.run(
            f"blastp -query {tmp_file} -db data/drosophila_refs/d_melanogaster.fasta -out {tmp_fileout} -outfmt 5",
            shell=True,
        )

        dmel_genes = {}
        with open(tmp_fileout, "rb") as result_stream:
            for blast_record in Blast.parse(result_stream):
                if len(blast_record) == 0:
                    continue
                src_gene = blast_record.query.description
                if src_gene in dmel_genes:
                    continue
                hit = blast_record[0].target.id
                dmel_genes[src_gene] = hit

        os.remove(tmp_file)
        os.remove(tmp_fileout)

        for gene in gene_names:
            if gene not in dmel_genes:
                dmel_genes[gene] = ""
        dmel_genes = pd.Series(dmel_genes)

        return dmel_genes

    def annotate_genes_with_dmel(species_code, genes):
        print("Get protein sequences...")
        protein_sequenced = get_protein_sequences(species_code, genes)

        print("BLASTing to Dmel...")
        blast_records = blast_genes_dmel(protein_sequenced)
        return blast_records

    dmel_genesd = {}
    for species_code in ["imin", "pnit", "roki", "mdar"]:
        print(f"Looking for soldier muscle markers for species {species_code}...")
        dmel_genes = annotate_genes_with_dmel(
            species_code,
            degsd[species_code].index[:10],
        )
        tmp = []
        dmel_genes.index.name = "original"
        dmel_genes = dmel_genes.to_frame(name="dmel").reset_index()
        dmel_genes["up_in"] = "soldier"
        tmp.append(dmel_genes)

        dmel_genes = annotate_genes_with_dmel(
            species_code,
            degsd[species_code].index[-10:][::-1],
        )
        dmel_genes.index.name = "original"
        dmel_genes = dmel_genes.to_frame(name="dmel").reset_index()
        dmel_genes["up_in"] = "king"
        tmp.append(dmel_genes)

        dmel_genes = pd.concat(tmp, axis=0)
        dmel_genes["species"] = species_code

        dmel_genesd[species_code] = dmel_genes
    dmel_genes = pd.concat(dmel_genesd.values())

    plt.ion()
    plt.show()
