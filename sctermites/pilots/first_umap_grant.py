# vim: fdm=indent
'''
author:     Fabio Zanini
date:       13/02/24
content:    Make a figure for the ARC DP grant EOI.
'''
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata


if __name__ == '__main__':

    print('Load data')
    fn = '../data/initial_4_samples/combined_caste_termites_znev.h5ad'
    adata = anndata.read_h5ad(fn)
    adata.obs['sample#'] = adata.obs_names.str.split('-', expand=True).get_level_values(1)
    adata.obs['caste'] = adata.obs['sample#'].map({
        '1': 'king',
        '2': 'queen',
        '3': 'soldier',
        '4': 'worker',
    })
    palette = {'king': 'steelblue', 'queen': 'pink', 'worker': 'grey', 'soldier': 'orange'}

    adata.obs['caste_king'] = adata.obs['caste'].map({
        'king': 'K',
        'queen': 'Q/W/S',
        'soldier': 'Q/W/S',
        'worker': 'Q/W/S',
    })
    palette = {'K': 'tomato', 'Q/W/S': (0.6, 0.6, 0.6, 0.01)}

    plt.ion()
    fig, ax = plt.subplots(figsize=(3.2, 3.1))
    sc.pl.umap(
        adata,
        color=['caste_king'], size=15, add_outline=True, ax=ax,
        palette=palette,
        title='',
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.text(0.05, 0.95, f'{adata.shape[0]} nuclei', va='top', ha='left', transform=ax.transAxes)
    ax.text(0.82, 0.03, 'King-only\ncell type', va='bottom', ha='right', transform=ax.transAxes)

    hs = [
        ax.scatter([], [], color=palette['K']),
        ax.scatter([], [], color=palette['Q/W/S'][:3]),
    ]
    ax.legend(
        hs, ['King', 'Q/W/S'],
        loc='upper right', frameon=False,
    )
    fig.tight_layout()
    fig.savefig('../figures/umap_4_initial_samples_for_DP.png', dpi=300)
    plt.show()
