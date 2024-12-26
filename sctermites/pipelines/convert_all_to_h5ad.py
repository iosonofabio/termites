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
            out_fn = fdn[len("run_count"):]
        else:
            continue
        print(fdn)
        h5_file_10x = count_data_fdn / fdn / "outs" / "filtered_feature_bc_matrix.h5"
        adata = sc.read_10x_h5(h5_file_10x)

        adata.write(h5ad_fdn / f"{out_fn}.h5ad")
