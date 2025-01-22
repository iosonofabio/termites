"""
Convet all termite single cell data to more convenient h5ad files.
"""


import os
import sys
import pathlib
import glob
import subprocess as sp
import anndata
import scanpy as sc


data_fdn = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/")


if __name__ == "__main__":

    count_data_fdn = data_fdn / "cell_ranger/matrices"
    count_subfdn = os.listdir(count_data_fdn)
    print(count_subfdn)

    h5ad_fdn = data_fdn / "h5ads"

    for fdn in count_subfdn:
        if fdn.startswith("run_count_"):
            out_fn = fdn[len("run_count_"):]
            out_fn = h5ad_fdn / f"{out_fn}.h5ad"
        else:
            continue

        print(fdn)
        if out_fn.exists():
            print("  Already exists, skipping")
            continue

        h5_file_10x = count_data_fdn / fdn / "outs" / "filtered_feature_bc_matrix.h5"
        try:
            adata = sc.read_10x_h5(h5_file_10x)
        except FileNotFoundError:
            print("  WARNING! No 10x data found, skipping")
            continue

        adata.write(out_fn)
