"""Train (i.e. run) SATURN on the termites/fly data.


This must be run inside the Python 3.10 saturn conda environment: source ~/miniconda3/bin/activate && conda activate saturn

This comes roughly from here:

https://github.com/snap-stanford/SATURN/blob/main/Vignettes/frog_zebrafish_embryogenesis/Train%20SATURN.ipynb
"""
import os
import pathlib
import numpy as np
import pandas as pd
import subprocess as sp


# NOTE: this is the table with the closest known genomes as of the time of writing. In the "docs" folder there is
# genomes_termite_data_usb_scRNA.ods
sample_dict = {
    "dmel": ["dmel.h5ad", "Dmel_gene_all_esm1b.pt"],
    "znev": ["znev.h5ad", "Znev_gene_all_esm1b.pt"],
    "roki": ["roki.h5ad", "Rfla_gene_all_esm1b.pt"],
    "ofor": ["ofor.h5ad", "Ofor_gene_all_esm1b.pt"],
    "mdar": ["mdar.h5ad", "Mdar_gene_all_esm1b.pt"],
    "hsjo": ["hsjo.h5ad", "Hsjo_gene_all_esm1b.pt"],
    "gfus": ["gfus.h5ad", "Gfus_gene_all_esm1b.pt"],
    "cpun": ["cpun.h5ad", "Punk_gene_all_esm1b.pt"],
    "imin": ["imin.h5ad", "Isch_gene_all_esm1b.pt"],
    "cfor": ["cfor.h5ad", "Cges_gene_all_esm1b.pt"],
    "nsug": ["nsug.h5ad", "Ncas_gene_all_esm1b.pt"],
}


def merge_caste_h5ad(input_fdn, output_fdn):
    import glob
    import anndata

    os.makedirs(output_fdn, exist_ok=True)

    for species, datum in sample_dict.items():
        print(species)
        h5ad_fn = output_fdn / datum[0]

        adatas = []
        fns = glob.glob(str(input_fdn) + f"/{species}*.h5ad")
        if len(fns) == 0:
            print(f"  WARNING! No files found for {species}")
            continue

        for fn in fns:
            caste = pathlib.Path(fn).stem.split("_")[1]
            if caste == "roauh":
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
        else:
            adata = anndata.concat(adatas, label="sample")
            adata.obs_names_make_unique()
            adata.var = adatas[0].var
            print(adata.obs)

        adata.write(h5ad_fn)


if __name__ == "__main__":

    embeddings_summary_fdn = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/saturn_data/esm_embeddings_summaries/")
    output_fdn = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/saturn_data/output")

    # Merge h5ad files across castes for each species (with annotation)
    h5ad_input_fdn = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/h5ads")
    h5ad_fdn = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/saturn_data/h5ad_by_species")
    merge_caste_h5ad(h5ad_input_fdn, h5ad_fdn)

    # Build the CSV used by SATURN to connect the species
    saturn_csv_fn = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/saturn_data/in_csv.csv")
    sample_dict_absolute = {key: [f"{h5ad_fdn}/{p0}", f"{embeddings_summary_fdn}/{p1}"] for key, (p0, p1) in sample_dict.items()}
    df = pd.DataFrame.from_records(sample_dict_absolute).T.reset_index().rename(columns={0: "path", 1: "embedding_path", "index": "species"})

    # Only use drosophila if available
    dmel_h5ad_fn = h5ad_fdn / df.loc[df["species"] == "dmel", "path"].values[0]
    if not dmel_h5ad_fn.exists():
        print("WARNING! No Drosophila h5ad data available, skipping and using znev SAMAP as reference")
        df = df.loc[df["species"] != "dmel"]

    df.to_csv(saturn_csv_fn, index=False)

    # Run SATURN
    script_fn = pathlib.Path(__file__).parent.parent.parent.parent / "software" / "SATURN" / "train-saturn.py"
    centroids_fn = output_fdn / "centroids.pkl"  # This is temp output to speed up later iterations (kmeans is costly, apparently)
    scoring_maps_fn = output_fdn / "scoring_cell_type_maps.csv"  # This is temp output to speed up later iterations (kmeans is costly, apparently)

    call = [
        "python",
        str(script_fn),
        "--in_data",
        str(saturn_csv_fn),
        "--in_label_col=cell_type",
        "--num_macrogenes=2000",
        "--hv_genes=8000",
        f"--centroids_init_path={centroids_fn}",
        "--score_adata",
        f"--ct_map_path={scoring_maps_fn}",  # This is output related to the scoring
        f"--work_dir={output_fdn}",  # This is general output
    ]
    print(" ".join(call))
    sp.run(" ".join(call), shell=True, check=True)
