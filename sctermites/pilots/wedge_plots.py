import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Example for a single gene
    fraction_exp = pd.DataFrame.from_dict(
        {
            "worker": [0.3, 0.1, 0.8, 0.3, 0.9],
            "queen": [0.2, 0, 0.4, 0, 0.95],
            "king": [0.1, 0.7, 0.2, 0, 0.97],
            "soldier": [0.7, 0, 0, 0.7, 1.0],
        },
    ).T
    fraction_exp.columns = ["gene1", "gene2", "gene3", "gene4", "housekeeping"]
    fraction_exp = fraction_exp[["housekeeping", "gene1", "gene2", "gene3", "gene4"]]

    angles = {
        "worker": {
            "center": np.pi,
            "span": np.pi,
        },
        "soldier": {
            "center": np.pi * (0.5 - 1.0 / 6),
            "span": np.pi / 3,
        },
        "queen": {
            "center": 0,
            "span": np.pi / 3,
        },
        "king": {
            "center": -np.pi * (0.5 - 1.0 / 6),
            "span": np.pi / 3,
        },
    }
    radial_pad = 0.07
    h_pad = radial_pad * 2 + 0.5
    v_pad = radial_pad * 2 + 0.1
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(5, 2))
    for ig, gene in enumerate(fraction_exp.columns):
        for caste in fraction_exp.index:
            angle = angles[caste]["center"]
            span = angles[caste]["span"]
            radius = fraction_exp.loc[caste, gene]
            wedge = plt.matplotlib.patches.Wedge(
                (
                    ig * (2 + h_pad) + radial_pad * np.cos(angle),
                    radial_pad * np.sin(angle),
                ),
                radius,
                np.degrees(angle - span / 2),
                np.degrees(angle + span / 2),
                facecolor=cmap(radius),
                edgecolor="black",
            )
            ax.add_patch(wedge)
    for caste in fraction_exp.index:
        angle = angles[caste]["center"]
        radius = 0.7
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        ax.text(
            x + 0.2 * (x < 0),
            y,
            caste[0].upper(),
            ha="center",
            va="center",
        )

    ax.set_xlim(-1.2, 1.2 + (len(fraction_exp.columns) - 1) * (2 + h_pad))
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks(np.arange(len(fraction_exp.columns)) * (2 + h_pad))
    ax.set_xticklabels(fraction_exp.columns, rotation=90)
    ax.set_yticks([])
    fig.tight_layout()

    plt.ion()
    plt.show()
