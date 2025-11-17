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
from Bio import Phylo
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import iplotx as ipx


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
species_dict_inv = {v: k for k, v in species_full_dict.items()}

higher_termites = [
    "ofor",
    "pnit",
]

additional_cell_typed = {
    "znev_14": "znev_14 (hemocyte)",
    "znev_13": "znev_13 (upper_digestive)",
    "znev_16": "znev_16 (glial)",
    "znev_4": "znev_4 (tracheo/sperm)",
    "znev_18": "znev_18 (ductal/tendon)",
}


def load_termite_tree():
    """Load the termite species tree."""
    tree_fn = "data/phylogeny/tree.tre.nexus"
    with open(tree_fn, "r") as f:
        tree = Phylo.read(f, "nexus")

    # Rename all leaves to remove quotation marks
    for leaf in tree.get_terminals():
        leaf.name = leaf.name.strip("'")

    # Rename spelling mistakes
    for leaf in tree.get_terminals():
        if leaf.name == "Hodotermopsis sjosdteti":
            leaf.name = "Hodotermopsis sjostedti"

    # Add species codes to leaves
    for leaf in tree.get_terminals():
        leaf.species_code = species_dict_inv[leaf.name]

    # Add Roki with some approximate distanced from speratus
    for parent in tree.get_nonterminals():
        for i, child in enumerate(parent.clades):
            if hasattr(child, "species_code") and child.species_code == "rspe":
                rspe = child
                break
        else:
            continue

        bl = rspe.branch_length
        bl_steal = 0.05 * bl
        bl_left = bl - bl_steal

        rspe.branch_length = bl_steal
        roki = Phylo.Newick.Clade(
            branch_length=bl_steal, name="Reticulitermes okinawus"
        )
        roki.species_code = "roki"

        reti_parent = Phylo.Newick.Clade(branch_length=bl_left)
        reti_parent.clades.append(rspe)
        reti_parent.clades.append(roki)
        parent.clades[i] = reti_parent
        break

    # Add ofor, which is a higher termite like pnit
    for parent in tree.get_nonterminals():
        for i, child in enumerate(parent.clades):
            if hasattr(child, "species_code") and child.species_code == "pnit":
                pnit = child
                break
        else:
            continue

        bl = pnit.branch_length
        bl_steal = 0.4 * bl
        bl_left = bl - bl_steal
        ofor = Phylo.Newick.Clade(
            branch_length=bl_steal,
            name="Odontotermes formosansus",
        )
        ofor.species_code = "ofor"
        ofor_parent = Phylo.Newick.Clade(branch_length=bl_left)
        ofor_parent.clades.append(ofor)
        ofor_parent.clades.append(pnit)
        parent.clades[i] = ofor_parent
        break

    # Add drosophila
    dmel = Phylo.Newick.Clade(
        branch_length=2.5,
        name="Drosophila melanogaster",
    )
    dmel.species_code = "dmel"
    tree.root.branch_length = 1.0
    new_root = Phylo.Newick.Clade()
    new_root.branch_length = 0.02
    new_root.clades.append(dmel)
    new_root.clades.append(tree.root)
    tree.root = new_root

    # Ladderize
    tree.ladderize()

    # Swap higher termites to be last
    # Swap cfor and Reticulitermes
    for parent in tree.get_nonterminals():
        if ofor_parent in parent.clades:
            parent.clades = parent.clades[::-1]

        if reti_parent in parent.clades:
            parent.clades = parent.clades[::-1]

    return tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savefig", action="store_true", help="Save the figures")
    args = parser.parse_args()

    print("Loading annotated h5ad with all termite species...")
    h5ad_fn = "data/saturn_output/final_adata_with_umap_and_znev-based_annotations_no_wrong_meta.h5ad"
    adata = anndata.read_h5ad(h5ad_fn)

    n_cts = (
        adata.obs.groupby(["species", "caste", "cell_type_znev_based"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    frac_cts = n_cts.div(n_cts.sum(axis=1), axis=0)

    if False:
        print("Check out most abundant cell types...")
        palette = dict(
            zip(adata.obs["species"].cat.categories, adata.uns["species_colors"])
        )
        cell_types_abu = adata.obs["cell_type_znev_based"].value_counts().index[:10]
        for cell_type in cell_types_abu:
            print(f"Select {cell_type}...")
            frac_ct = frac_cts[cell_type]

            print("Exclude outgroups")
            frac_ct = frac_ct.drop(
                index=[("dmel", "n/a"), ("cpun", "n/a")], errors="ignore"
            )

            print("Proportion across castes...")
            prop_ct_castes = frac_ct.unstack()[["soldier", "king"]].fillna(0)
            colors = [palette[sp] for sp in prop_ct_castes.index.get_level_values(0)]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(
                100 * prop_ct_castes["soldier"],
                100 * prop_ct_castes["king"],
                s=100,
                c=colors,
            )
            ax.set_xlabel(f"% {cell_type} in soldier")
            ax.set_ylabel(f"% {cell_type} in king")
            ax.grid(True)
            ax.set_title(cell_type)
            vmax = prop_ct_castes.values.max() * 100 * 1.1
            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
            ax.plot([0, vmax], [0, vmax], ls="--", color="black", lw=2)
            texts = []
            for species_code, row in prop_ct_castes.iterrows():
                t = ax.text(row["soldier"] * 100, row["king"] * 100, species_code)
                texts.append(t)
            adjust_text(texts)
            fig.tight_layout()

        # The most interesting seem to be:
        # - fat cells, often higher in king (except for very lean species)
        # - neuron 2, vice versa clearly favours soldier
        # - znev_14 and _18, mostly favour soldier

    print("Plot the abundances against the tree")
    tree = load_termite_tree()

    castes = ["soldier", "worker", "king", "queen", "n/a"]
    palette = {
        "king": "#1f77b4",
        "queen": "#ff7f0e",
        "soldier": "#2ca05c",
        "worker": "#962758",
        "n/a": "#7f7f7f",
    }
    caste_by_species = {
        key: list(np.unique(val["caste"]))
        for key, val in adata.obs.groupby("species", observed=True)
    }

    cell_type_groups = [
        [
            "epithelial",
            "muscle cell",
            "neuron",
            "sensory neuron",
            "gustatory neuron",
            "stem cell",
            "fat cell",
            "T1",
            "oenocyte",
            "znev_missing_3",
            "znev_missing_2",
        ],
    ]
    other_cell_types = [
        ct
        for ct in adata.obs["cell_type_znev_based"].cat.categories
        if ct not in cell_type_groups[0]
    ]
    other_cell_types.sort()
    cell_type_groups.append(other_cell_types)

    for igroup, cell_types in enumerate(cell_type_groups):
        fig, axs = plt.subplots(
            1,
            1 + len(cell_types),
            figsize=(5 + 2.4 * len(cell_types), 6),
            gridspec_kw=dict(width_ratios=[4] + [2.7] * len(cell_types)),
            sharey=True,
        )
        ta = ipx.tree(
            tree,
            ax=axs[0],
            leaf_deep=True,
            title="% cells in that type by species and caste",
        )

        # Style higher termites differently
        leaf_higher = [
            tree_leaf
            for tree_leaf in tree.get_terminals()
            if tree_leaf.species_code in higher_termites
        ]
        ta.style_subtree(leaf_higher, edge_color="tomato", lw=2)

        for iax, cell_type in enumerate(cell_types, 1):
            frac_ct = frac_cts[cell_type]
            vmin = 0
            vmax = frac_ct.values.max()
            cmap = mpl.colormaps["Reds"]
            ax = axs[iax]
            ax.set_title(additional_cell_typed.get(cell_type, cell_type))
            ax.set_yticks(np.arange(len(tree.get_terminals())))
            ax.set_yticklabels([leaf.name for leaf in tree.get_terminals()])
            ax.set_xlim(-0.5, len(castes) - 0.5)
            ax.set_ylim(-0.5, len(tree.get_terminals()) - 0.5)
            ax.set_xticks(np.arange(len(castes)))
            ax.set_xticklabels(castes, rotation=90, ha="center")
            ax.axhline(7.5, ls="--", color="black", alpha=0.5, lw=1.5)
            ax.axhline(9.5, ls="--", color="black", alpha=0.5, lw=1.5)
            for iy, leaf in enumerate(tree.get_terminals()):
                species_code = leaf.species_code
                for ix, caste in enumerate(castes):
                    if caste not in caste_by_species[species_code]:
                        continue
                    if (species_code, caste) in frac_ct.index:
                        abu = frac_ct.loc[(species_code, caste)]
                    else:
                        abu = 0
                    fc = cmap((abu - vmin) / (vmax - vmin))
                    ax.add_patch(
                        plt.Rectangle(
                            (ix - 0.45, iy - 0.45),
                            0.9,
                            0.9,
                            edgecolor=palette[caste],
                            facecolor=fc,
                            lw=2,
                        )
                    )
                    tc = "black" if abu < (vmax - vmin) / 2 else "white"
                    ax.text(
                        ix, iy, f"{100 * abu:.0f}", ha="center", va="center", color=tc
                    )
        axs[0].invert_yaxis()
        fig.tight_layout(h_pad=0)

        if args.savefig:
            fig.savefig(
                f"figures/cell_type_abundance_tree_group_{igroup + 1}.svg",
            )
            fig.savefig(
                f"figures/cell_type_abundance_tree_group_{igroup + 1}.png",
                dpi=300,
            )

    plt.ion()
    plt.show()
