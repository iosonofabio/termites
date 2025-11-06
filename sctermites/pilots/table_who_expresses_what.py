import anndata
import numpy as np
import pandas as pd


if __name__ == "__main__":
    adata = anndata.read_h5ad(
        "data/initial_4_samples/combined_caste_termites_znev.h5ad"
    )

    # Reconstruct caste from library hints, Catherine has the real annotation here

    # Mock the cell type column, we should have a proper one here
    adata.obs["cell_type"] = pd.Categorical(adata.obs["leiden"].astype(str))
