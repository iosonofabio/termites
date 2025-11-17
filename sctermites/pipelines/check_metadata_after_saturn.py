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
    parser.add_argument("--savefig", action="store_true", help="Save the figures")
    args = parser.parse_args()

    print("Loading annotated h5ad with all termite species...")
    h5ad_fn = "data/saturn_output/final_adata_with_umap_and_znev-based_annotations.h5ad"
    adata = anndata.read_h5ad(h5ad_fn)
    obs = adata.obs

    print("Load metadata from individual species and castes...")
    obsd = {}

    print("Loading annotated h5ad with Znev cell types...")
    h5ad_znev_fn = "data/znev/combined_no_norm_clustered_new_znev.h5ad"
    adata_znev = anndata.read_h5ad(h5ad_znev_fn, backed="r")
    for caste in ["soldier", "worker", "king", "queen"]:
        print(f"znev {caste}")
        mask = adata_znev.obs["caste"] == caste
        obsd[("znev", caste)] = adata_znev.obs[mask]

    caste_coded = {
        "sol": "soldier",
        "king": "king",
        "roach": "n/a",
    }
    for fn in pathlib.Path("data/all_species/h5ads/").glob("*.h5ad"):
        species_code = fn.stem.split("_")[0]
        caste_code = fn.stem.split("_")[1].split(".")[0]

        caste = caste_coded[caste_code]

        print(f"Loading {species_code} from {fn}...")
        adata_sp = anndata.read_h5ad(fn, backed="r")
        obsd[(species_code, caste)] = adata_sp.obs

    print("Checking for missing cells...")
    for (species, caste), obsi in obsd.items():
        nmiss = (~obsi.index.isin(obs.index)).sum()
        print(f"{species} {caste}: {nmiss}")
    print("... there are a few")

    print("Check for duplicate cell IDs in saturn output...")
    print(obs.index.duplicated().sum())
    print("... none found.")

    # TODO: Check also drosophila

    print("Check species and caste combinations...")
    wrong_meta = []
    for (species, caste), obsi in obsd.items():
        obs_subset = obs.loc[obs.index.isin(obsi.index)]
        print(f"Should be all: {species} {caste}...")
        tab = (
            obs_subset.groupby(["species", "caste"], observed=True)
            .size()
            .sort_values(ascending=False)
        )
        frac_off = 100.0 * (tab.sum() - tab.get((species, caste), 0)) / tab.sum()
        print(tab)
        print(f"Percent off: {frac_off:.0f}%")

        # Track the ones that should be this species/caste but are not
        obs_off = obs_subset[
            (obs_subset["species"] != species) | (obs_subset["caste"] != caste)
        ].index.tolist()
        wrong_meta.extend(obs_off)

        # Also track the ones that are assigned to this species/caste but should not be
        obs_off = obs.loc[~obs.index.isin(obsi.index)]
        obs_off = obs_off[
            (obs_off["species"] == species) & (obs_off["caste"] == caste)
        ].index.tolist()
        wrong_meta.extend(obs_off)
    print("... 1-2%. This is not great but we can exclude these for now.")

    print("Also exclude cells with caste 'n/a'... unless they are roach or fly")
    obs_na_caste = obs.loc[
        (obs["caste"] == "n/a") & ~(obs["species"].isin(["dmel", "cpun"]))
    ].index.tolist()
    wrong_meta.extend(obs_na_caste)

    print(
        "Also filter cells that come from a caste that was not sequenced from that species..."
    )
    species_by_casted = defaultdict(set)
    for species, caste in obsd.keys():
        species_by_casted[species].add(caste)
    for species, castes_this_species in species_by_casted.items():
        obs_off = obs.loc[
            (obs["species"] == species) & ~(obs["caste"].isin(castes_this_species))
        ].index.tolist()
        print(
            f"{len(obs_off)} cells from {species} with wrong castes not in {castes_this_species}"
        )
        wrong_meta.extend(obs_off)

    print("Remove cells with wrong metadata...")
    adata.obs["wrong_meta_after_saturn"] = adata.obs_names.isin(wrong_meta)
    adata = adata[~adata.obs["wrong_meta_after_saturn"]]

    print("Save new h5ad without wrong metadata cells...")
    h5ad_fn = "data/saturn_output/final_adata_with_umap_and_znev-based_annotations_no_wrong_meta.h5ad"
    adata.write(h5ad_fn)
