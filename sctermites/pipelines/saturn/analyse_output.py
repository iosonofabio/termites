"""Analyse the output of SATURN

This script requires anndata and scanpy. One way to do that is to use the SATURN conda environment:

source ~/miniconda3/bin/activate && conda activate saturn

"""

import os
import sys
import pathlib
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
import anndata
import torch
import scanpy as sc
import atlasapprox as aa
from Bio import Phylo
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# NOTE: this is the table with the closest known genomes as of the time of writing. In the "docs" folder there is
# genomes_termite_data_usb_scRNA.ods
sample_dict = {
    # Perfect genome matches
    "dmel": ["d_melanogaster.h5ad", "Dmel_gene_all_esm1b.pt"],
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

species_full_dict = {
    "dmel": "Drosophila melanogaster",
    "znev": "Zootermopsis nevadensis",
    "ofor": "Odontotermes formosansus",
    "mdar": "Mastotermes darwiniensis",
    "hsjo": "Hodotermopsis sjostedti",
    "gfus": "Glyptotermes fuscus",
    "imin": "Incisitermes minor",
    "cfor": "Coptotermes Formosanus",
    "nsug": "Neotermes sugioi",
    "pnit": "Pericapritermes nitobei",
    "cpun": "Cryptocercus punctulatus",
    "roki": "Reticulitermes okinawus",
    "rspe": "Reticulitermes speratus",
}


def compress(adata, ct_col):
    cell_types = adata.obs[ct_col].cat.categories
    nvar = adata.n_vars
    nct = len(cell_types)
    avg = np.zeros((nct, nvar), np.float32)
    frac = np.zeros((nct, nvar), np.float32)
    for i, ct in enumerate(cell_types):
        avg[i] = adata[adata.obs[ct_col] == ct].X.mean(axis=0)
        frac[i] = (adata[adata.obs[ct_col] == ct].X > 0).mean(axis=0)

    adatac = anndata.AnnData(
        X=avg,
        layers={"fraction": frac},
        obs=pd.DataFrame(index=cell_types),
        var=adata.var.copy(),
    )
    return adatac


def get_markers(approx, name, number=10):
    vec = np.asarray(approx[name].layers["fraction"]).ravel()
    other_idx = [x for x in approx.obs_names if x != name]
    other = np.asarray(approx[other_idx].layers["fraction"])

    diff_min = pd.Series(
        (vec - other).min(axis=0),
        index=approx.var_names,
    )
    cands = diff_min.nlargest(number)
    cands = cands[cands > 0]
    return cands


def get_embeddings_esm1b(species, genes=None, chain=True):
    if species in species_full_dict:
        species_cap = species[0].upper() + species[1:]
        fn = f"/mnt/data/projects/termites/data/sc_termite_data/saturn_data/esm_embeddings_summaries/{species_cap}_gene_all_esm1b.pt"
    else:
        fn = f"/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/esm_embeddings_summaries/{species}_gene_all_esm1b.pt"

    fn = pathlib.Path(fn)
    if not fn.exists():
        raise IOError(f"Embeddings file not found for species {species}: {fn}")

    emb_dict = torch.load(fn)

    if genes is not None:
        emb_dict = {emb_dict[g] for g in genes}

    if chain:
        genes = np.array(list(emb_dict.keys()))
        X = torch.stack(list(emb_dict.values()))
        emb_dict = {"genes": genes, "X": X}

    return emb_dict


def search_homologs(emb_dict1, emb_dict2, k=1):
    print("Searching for homologs")
    if "genes" in emb_dict1:
        genes1 = emb_dict1["genes"]
        X1 = emb_dict1["X"]
    else:
        genes1 = np.array(list(emb_dict1.keys()))
        X1 = torch.stack(list(emb_dict1.values()))

    if "genes" in emb_dict2:
        genes2 = emb_dict2["genes"]
        X2 = emb_dict2["X"]
    else:
        genes2 = np.array(list(emb_dict2.keys()))
        X2 = torch.stack(list(emb_dict2.values()))

    cdis = torch.cdist(X1, X2, p=2)
    dmin = cdis.topk(dim=1, largest=False, k=k)
    genes_tgt = genes2[dmin.indices.numpy().ravel()]
    return genes_tgt


def check_homologs_expressors(species, genes):
    print("Get highest expressors of homologs")
    api = aa.API()
    number = 5
    res = {}
    for gene in genes:
        highest_exp = api.highest_measurement(
            organism=species,
            feature=gene,
            number=number,
        )
        res[gene] = highest_exp
    res = pd.concat(res.values()).reset_index()["celltype"].value_counts()

    # Null model
    print("Get highest expressors of null genes")
    genes_random = np.random.choice(genes, number, replace=False)
    res_rand = {}
    for gene in genes_random:
        highest_exp = api.highest_measurement(
            organism=species,
            feature=gene,
            number=number,
        )
        res_rand[gene] = highest_exp
    res_rand = pd.concat(res_rand.values()).reset_index()["celltype"].value_counts()
    df = (
        res.to_frame("res")
        .join(res_rand.to_frame("null"), how="outer")
        .fillna(0)
        .astype(int)
        .nlargest(5, "res")
    )
    df = df.loc[df["res"] > 2 * df["null"]]
    if df.shape[0] == 0:
        return None
    if df.shape[0] == 1:
        return df.index[0]
    if df["res"].iloc[0] > df["res"].iloc[1] + 2:
        return df.index[0]
    return df.index[:2]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--savefig", action="store_true", help="Save the figures")
    args = parser.parse_args()

    output_fdn = pathlib.Path(
        "/mnt/data/projects/termites/data/sc_termite_data/saturn_data/output_1700_3700"
    )
    saturn_h5ad = output_fdn / "saturn_results" / "final_adata_with_umap.h5ad"
    has_metadata = True
    if not saturn_h5ad.exists():
        has_metadata = False
        saturn_h5ad = output_fdn / "saturn_results" / "final_adata.h5ad"

    print("Read phylogeny")
    tree_fn = "../data/trees/tree_from_tom.nwk"
    tree = Phylo.read(tree_fn, "newick")
    species_order = [leaf.name.lower() for leaf in tree.get_terminals()]

    print("Read h5ad")
    adata = anndata.read_h5ad(saturn_h5ad)

    if not has_metadata:
        print("Add caste information")
        adata.obs["caste"] = ""
        separate_h5ad_fdn = pathlib.Path(
            "/mnt/data/projects/termites/data/sc_termite_data/saturn_data/h5ad_by_species"
        )
        for species, datum in sample_dict.items():
            print(species)
            h5ad_fn = separate_h5ad_fdn / datum[0]
            if not h5ad_fn.exists():
                continue
            adatas = anndata.read_h5ad(h5ad_fn)
            if species in ("dmel", "cpun"):
                adatas.obs["caste"] = "n/a"
            cell_ids_species = adata.obs_names[adata.obs["species"] == species]
            caste_species = adatas.obs.loc[cell_ids_species, "caste"]
            adata.obs.loc[cell_ids_species, "caste"] = caste_species
            del adatas
        adata.obs["caste"] = adata.obs["caste"].astype(str).replace("sol", "soldier")
        adata.obs["caste"] = pd.Categorical(adata.obs["caste"])
        __import__("gc").collect()

        print("Now we can make obs unique")
        adata.obs_names_make_unique()

        # NOTE: no normalisation needed in embedding space

        print("PCA")
        sc.pp.pca(adata)

        print("KNN")
        sc.pp.neighbors(adata)

        print("UMAP")
        sc.tl.umap(adata)

        adata.obs["species_full"] = adata.obs["species"].map(species_full_dict)
        adata.obs["labels_only_znev"] = adata.obs["labels2"].astype(str)
        adata.obs.loc[adata.obs["species"] != "znev", "labels_only_znev"] = (
            "unknown (not znev)"
        )
        adata.obs["labels_only_znev"] = pd.Categorical(adata.obs["labels_only_znev"])
        adata.obs["labels_only_dmel"] = adata.obs["labels2"].astype(str)
        adata.obs.loc[adata.obs["species"] != "dmel", "labels_only_dmel"] = (
            "unknown (not dmel)"
        )
        adata.obs["labels_only_dmel"] = pd.Categorical(adata.obs["labels_only_dmel"])

        print("Clustering")
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=0.2)

        print("Store AnnData with annotations, UMAP, etc.")
        adata.write(saturn_h5ad.parent / "final_adata_with_umap.h5ad")
        print("Stored")

    if False:
        print("Try to figure out what each cloud could be based on znev annotation")
        print("  Read znev h5ad for reference")
        znev_h5ad_fn = pathlib.Path(
            "/mnt/data/projects/termites/data/sc_termite_data/saturn_data/h5ad_by_species/znev.h5ad",
        )
        adata_znev = anndata.read_h5ad(znev_h5ad_fn)
        approx_znev = compress(adata_znev, "cell_type")
        emb_dict_human = get_embeddings_esm1b("h_sapiens")
        emb_dict_fly = get_embeddings_esm1b("d_melanogaster")
        emb_dict_znev = get_embeddings_esm1b("znev", chain=False)
        for clu in adata.obs["leiden"].cat.categories:
            znev_label_counts = adata.obs.loc[
                (adata.obs["species"] == "znev") & (adata.obs["leiden"] == clu),
                "ref_labels",
            ].value_counts()
            if znev_label_counts.sum() == 0:
                print("Cluster", clu, "has NO znev cells")
                continue
            if znev_label_counts.iloc[0] > 0.5 * znev_label_counts.sum():
                znev_label = znev_label_counts.index[0]
                pct = int(100 * znev_label_counts.iloc[0] / znev_label_counts.sum())
                print(f"Cluster {clu} is mostly Znev {znev_label} ({pct}%)")

                if not znev_label.isdigit():
                    continue

                markers = get_markers(approx_znev, znev_label)
                tmp_dict = {gene: emb_dict_znev[gene] for gene in markers.index}
                homologs = search_homologs(tmp_dict, emb_dict_fly, k=2)
                top_expressors = check_homologs_expressors("d_melanogaster", homologs)
                if top_expressors is None:
                    print(
                        f"  It is unclear what fly cell type expresses homologs of those markers"
                    )
                    continue
                if isinstance(top_expressors, str):
                    print(
                        f"  Fly cell type {top_expressors} is a likely candidate for this cluster"
                    )
                else:
                    print(
                        f"  Fly cell types {top_expressors[0]} and {top_expressors[1]} are somewhat candidates for this cluster"
                    )
                continue
            print("Cluster", clu, "is mixed:")
            for il, (znev_label, count) in enumerate(znev_label_counts.items()):
                if (count < 10) and (il != 0):
                    break
                print(f"  {znev_label}: {count}")

        sys.exit()

    print("Visualise")
    plt.ion()
    plt.close("all")

    print("Plot genome phylogeny")
    fig, ax = plt.subplots()
    Phylo.draw(tree, axes=ax)
    fig.tight_layout()
    if args.savefig:
        fig.savefig(
            "../figures/phylogeny_all_species.png",
            dpi=600,
        )

    # sc.pl.umap(adata, color="species_full", title="Species", add_outline=True, size=20)
    # fig = plt.gcf()
    # fig.set_size_inches(7, 5)
    # fig.tight_layout()

    palette = sns.color_palette(
        "husl", n_colors=len(adata.obs["labels_only_dmel"].cat.categories) - 1
    )
    palette.append((0.1, 0.1, 0.1, 0.001))
    fig2, ax = plt.subplots(figsize=(10, 5))
    sc.pl.umap(
        adata,
        color="labels_only_dmel",
        title="Cell Type",
        add_outline=True,
        size=20,
        palette=palette,
        groups=[
            x
            for x in adata.obs["labels_only_dmel"].cat.categories
            if "unknown" not in x
        ],
        na_color=palette[-1],
        ax=ax,
    )
    fig2.tight_layout()
    if args.savefig:
        fig2.savefig(
            "../figures/combined_umap_saturn_all_species_cell_type.png",
            dpi=600,
        )

    if False:
        print("Split Drosophila cell types in 6")
        palette = sns.color_palette(
            "husl", n_colors=len(adata.obs["labels_only_dmel"].cat.categories) - 1
        )
        palette.append((0.1, 0.1, 0.1, 0.001))
        plt.ioff()
        fig, axs = plt.subplots(3, 2, figsize=(13, 15), sharex=True, sharey=True)
        ct_colored = list(adata.obs["labels_only_dmel"].cat.categories)
        ct_colored.remove("unknown (not dmel)")
        for i, ax in enumerate(axs.ravel()):
            sc.pl.umap(
                adata,
                color="labels_only_dmel",
                title="Cell Type",
                add_outline=True,
                size=20,
                palette=palette,
                groups=ct_colored[i::6],
                na_color=palette[-1],
                ax=ax,
                show=False,
            )
            ax.set(xlabel="", ylabel="")
            if i != 0:
                ax.set(title="")
        fig.tight_layout()
        plt.ion()
        plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    sc.pl.umap(
        adata,
        color="leiden",
        title="Leiden clusters",
        add_outline=True,
        size=20,
        legend_loc="on data",
        ax=ax,
    )
    if args.savefig:
        fig.savefig(
            "../figures/combined_umap_saturn_all_species_leiden_clustering.png",
            dpi=600,
        )

    if False:
        fig3, axs = plt.subplots(2, 7, figsize=(21, 6), sharex=True, sharey=True)
        axs = axs.ravel()
        palette = {
            1: "tomato",
            0: (0.9, 0.9, 0.9, 0.001),
        }
        for species, ax in zip(species_order, axs):
            adata.obs["is_focal"] = pd.Categorical(
                (adata.obs["species"] == species).astype(int)
            )
            sc.pl.umap(
                adata,
                color="is_focal",
                title=species_full_dict[species],
                add_outline=True,
                size=20,
                ax=ax,
                legend_loc=None,
                palette=palette,
                groups=[1],
                na_color=palette[0],
            )
            ax.set(xlabel="", ylabel="")
        axs[-1].axis("off")
        fig3.tight_layout()
        if args.savefig:
            fig3.savefig("../figures/combined_umap_saturn_all_species", dpi=600)

    if False:
        palette = {
            1: tuple(list(mpl.colors.to_rgb("tomato")) + [0.01]),
            0: (0.9, 0.9, 0.9, 0.001),
        }
        fig4, axs = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True)
        for ax, caste in zip(axs, ["n/a", "king", "soldier", "queen", "worker"]):
            adata.obs["is_focal"] = pd.Categorical(
                (adata.obs["caste"] == caste).astype(int)
            )
            sc.pl.umap(
                adata,
                color="is_focal",
                title=caste,
                add_outline=True,
                size=20,
                ax=ax,
                legend_loc=None,
                palette=palette,
                groups=[1],
                na_color=palette[0],
            )
            ax.set(xlabel="", ylabel="")
        fig4.tight_layout()
        if args.savefig:
            fig4.savefig(
                "../figures/combined_umap_saturn_all_species_caste.png",
                dpi=600,
            )

    if True:
        print("Assign rough cell types based on drosophila")
        adatafly = adata[adata.obs["species"] == "dmel"]
        tmp = (
            adatafly.obs[["labels2", "leiden", "caste"]]
            .groupby(["leiden", "labels2"])
            .size()
            .unstack("labels2", fill_value=0)
        )
        clu_dict = {}
        for clu, row in tmp.iterrows():
            top = row.nlargest(5)
            topcum = top.cumsum()
            for it in range(len(topcum)):
                if topcum.iloc[it] > 0.4 * row.sum():
                    clu_dict[clu] = ", ".join(top.index[: it + 1])
                    break
        adata.obs["leiden_dmel"] = adata.obs["leiden"].map(clu_dict)
        adata.obs["leiden_dmel"] = pd.Categorical(adata.obs["leiden_dmel"])

        print("Compute cell type abundances in soldiers and outgroups")
        adata_sol = adata[adata.obs["caste"].isin(["soldier", "n/a"])]
        ct_abu_sol = (
            adata_sol.obs[["species", "leiden_dmel", "caste"]]
            .groupby(["species", "leiden_dmel"])
            .size()
            .unstack("leiden_dmel", fill_value=0)
            .loc[species_order]
        )
        ct_frac_sol = (1.0 * ct_abu_sol.T / ct_abu_sol.sum(axis=1)).T

        fig, ax = plt.subplots(figsize=(15, 3))
        nspe = ct_frac_sol.shape[0]
        nct = ct_frac_sol.shape[1]
        colors = sns.color_palette("husl", n_colors=nspe)
        for i, col in enumerate(ct_frac_sol.columns):
            ax.bar(
                np.linspace(i - 0.4, i + 0.4, nspe),
                ct_frac_sol[col],
                bottom=0,
                width=0.8 / nspe,
                color=colors,
            )
            if i != 0:
                ax.axvline(i - 0.5, color="black", lw=0.5, ls="--")
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(nspe)]
        ax.legend(
            handles,
            ct_frac_sol.index,
            title="Species",
            bbox_to_anchor=(1, 1),
            bbox_transform=ax.transAxes,
            loc="upper left",
            ncol=2,
        )
        ax.set_xlim(-0.5, nct - 0.5)
        ax.set_ylim(1e-4, 0.3)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(nct))
        ax.set_xticklabels(ct_frac_sol.columns, rotation=90)
        ax.set_ylabel("Fraction\nof cells")
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                "../figures/cell_type_abundance_inferred_from_dmel_soldier.png",
                dpi=600,
            )

        print("Message passing inference for cell types abundances across the tree")
        # Pass 1: up the tree
        for leaf in tree.get_terminals():
            leaf.fracs = leaf.mp_up = ct_frac_sol.loc[leaf.name.lower()]
        for node in tree.get_nonterminals(order="postorder"):
            node.mp_up = pd.Series(np.zeros(nct), index=ct_frac_sol.columns)
            for child in node.clades:
                node.mp_up += child.mp_up / child.branch_length
            node.mp_up /= sum(1.0 / child.branch_length for child in node.clades)
        # Pass 2: down the tree
        for node in tree.get_nonterminals(order="preorder"):
            if node == tree.root:
                node.fracs = tree.root.mp_up
            else:
                node.fracs = node.mp_down / node.branch_length
                for child in node.clades:
                    node.fracs += child.mp_up / child.branch_length
                node.fracs /= (
                    sum(1.0 / child.branch_length for child in node.clades)
                    + 1 / node.branch_length
                )
            for child in node.clades:
                child.mp_down = node.fracs

        # FIXME: actually, there's a bug above about avoiding the message from the same edge, plus
        # with such a small tree we can just listen to all evidence for each node
        nspe = len(species_order)
        # 1. Set the distances and abundances of the leaves (deltas)
        for leaf in tree.get_terminals():
            leaf.distances = pd.Series(-np.ones(nspe), index=species_order)
            leaf.distances.loc[leaf.name.lower()] = 0
            leaf.fracs = ct_frac_sol.loc[leaf.name.lower()]
        # 2. Set the distances to the hanging leaves, learning from the children
        for node in tree.get_nonterminals(order="postorder"):
            node.distances = pd.Series(-np.ones(nspe), index=species_order)
            for child in node.clades:
                idx = child.distances >= 0
                node.distances[idx] = child.distances[idx] + child.branch_length
        # NOTE: the root distances are just the depth of the leaves
        # 3. Set the distances for the non-hanging leaves, learning from the parent
        for node in tree.get_nonterminals(order="level"):
            for child in node.clades:
                idx = child.distances < 0
                child.distances[idx] = node.distances[idx] + child.branch_length
        # 4. Set the fractions of internal nodes as weighted averages of the evidence (leaves)
        for node in tree.get_nonterminals(order="level"):
            node.fracs = (ct_frac_sol.T / node.distances).T.sum(
                axis=0
            ) / node.distances.sum()

        # Set height and depth for each node, for plotting
        for node in tree.get_nonterminals(order="level") + tree.get_terminals():
            if node == tree.root:
                node.depth = 0
            for child in node.clades:
                child.depth = node.depth + child.branch_length
        for node in tree.get_terminals() + tree.get_nonterminals(order="postorder"):
            if node.is_terminal():
                node.height = species_order.index(node.name.lower())
            else:
                node.height = 0
                for child in node.clades:
                    node.height += child.height
                node.height /= len(node.clades)

        cmap = sns.diverging_palette(250, 15, s=75, l=40, center="dark", as_cmap=True)
        cmap2 = mpl.cm.get_cmap("viridis")
        fig, axs = plt.subplots(3, 7, figsize=(17.5, 7.5))
        axs = axs.ravel()
        for iax, (ct, ax) in enumerate(zip(ct_frac_sol.columns, axs)):
            scatter_data = defaultdict(list)
            for node in tree.get_nonterminals(order="level"):
                x0 = node.depth
                y0 = node.height
                for child in node.clades:
                    x1 = child.depth
                    y1 = child.height
                    xs = [x0, x0, x1]
                    ys = [y0, y1, y1]
                    diff = child.fracs[ct] - node.fracs[ct]
                    # diff = np.log(child.fracs[ct] + 1e-4) - np.log(
                    #    node.fracs[ct] + 1e-4
                    # )
                    diff = np.clip(np.clip(diff, -0.3, 0.3) / 0.6 + 0.5, 0, 1)
                    color = cmap(diff)
                    ax.plot(xs, ys, color=color, lw=3)
                    fr_clip = min(child.fracs[ct] * 2, 1.0)
                    scatter_data["x"].append(x1)
                    scatter_data["y"].append(y1)
                    # scatter_data["c"].append(cmap2(fr_clip))
                    scatter_data["s"].append(150 * fr_clip**0.5)
            ax.scatter(**scatter_data, color="black", zorder=10)
            ax.set_title(ct)
            ax.set_xticks([])
            if iax % 7 == 6:
                ax.set_yticks(np.arange(len(species_order)))
                ax.set_yticklabels(species_order, fontsize=10)
            else:
                ax.set_yticks([])
                ax.set_yticklabels([])
            ax.yaxis.set_ticks_position("right")
            ax.set_ylim(nspe - 0.5, -0.5)
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                "../figures/cell_type_abundance_change_across_tree_inferred_from_dmel_soldier.png",
                dpi=600,
            )

        print(
            "Plot relative distances in phylogeny and relative difference in abundance"
        )
        nct = ct_frac_sol.shape[1]
        distance_matrix = pd.DataFrame(
            {leaf.name.lower(): leaf.distances for leaf in tree.get_terminals()}
        )
        palette = dict(
            zip(ct_frac_sol.columns, sns.color_palette("husl", n_colors=nct))
        )
        data = []
        for ct in ct_frac_sol.columns:
            for i, spe1 in enumerate(species_order):
                for spe2 in species_order[:i]:
                    dis_phy = distance_matrix.at[spe1, spe2]
                    delta_abu = np.abs(
                        ct_frac_sol.loc[spe1, ct] - ct_frac_sol.loc[spe2, ct]
                    )
                    data.append(
                        {
                            "dis_phy": dis_phy,
                            "delta_frac": delta_abu,
                            "spe1": spe1,
                            "spe2": spe2,
                            "cell_type": ct,
                            "color": palette[ct],
                        }
                    )
        data = pd.DataFrame(data)
        from scipy.stats import pearsonr, spearmanr

        # Exclude drosophila for the stats, because cell type abundance is odd there
        data_for_stats = data.loc[(data["spe1"] != "dmel") & (data["spe2"] != "dmel")]
        r = pearsonr(data_for_stats["dis_phy"], data_for_stats["delta_frac"])
        rho = spearmanr(data_for_stats["dis_phy"], data_for_stats["delta_frac"])

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(
            data["dis_phy"],
            data["delta_frac"],
            c=data["color"],
            s=20,
            alpha=0.2,
            zorder=5,
        )
        sns.kdeplot(
            data,
            x="dis_phy",
            y="delta_frac",
            ax=ax,
            cmap="viridis",
            zorder=3,
        )
        ax.set_xlabel("Phylogenetic distance")
        ax.set_ylabel("$\\vert \Delta_f \\vert$")
        fig.tight_layout()
