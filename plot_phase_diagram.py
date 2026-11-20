import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
from matplotlib.legend_handler import HandlerTuple
from scipy.spatial import ConvexHull

from Des.cluster_expansion import list_all_energies

DATA_PATH = "phase_diagram_data.csv"

RED = "#cc0101"
YELLOW = "#f1ed00"
BLUE = "#0084FF"

def phase_diagram():
    df = pd.read_csv(DATA_PATH)
    df = df[df["Include in plot"]==True]

    # Split polarized and non polarized calculations
    spin_polarized_df = df[df["Spin polarized"]==True]
    non_spin_polarized_df = df[df["Spin polarized"]==False]

    fig, ax = plt.subplots(figsize=(16, 10))

    # Spin polarized data
    outer = ax.scatter(
        x=spin_polarized_df["Al fraction"],
        y=spin_polarized_df["Formation energy (Ry)"],
        s=60, color=RED, zorder=2, clip_on=False
    )
    inner = ax.scatter(
        x=spin_polarized_df["Al fraction"],
        y=spin_polarized_df["Formation energy (Ry)"],
        s=20, color=YELLOW, zorder=3, clip_on=False
    )

    # Non polarized data
    non_pol = ax.scatter(
        x=non_spin_polarized_df["Al fraction"],
        y=non_spin_polarized_df["Formation energy (Ry)"],
        s=60, color=RED, zorder=2, clip_on=False
    )

    # Convex hull (pure elements have E_form = 0 by definition)
    ref_x = [0.0, 100.0]
    ref_y = [0.0, 0.0]
    hull_x = np.concatenate([df["Al fraction"].to_numpy(), ref_x])
    hull_y = np.concatenate([df["Formation energy (Ry)"].to_numpy(), ref_y])

    # x is in %, y is in Ry: rescale x to a fraction so Qhull's tolerances behave
    points = np.column_stack([hull_x / 100, hull_y])
    ch = ConvexHull(points)

    # Keep only the lower facets: outward normal has a negative y-component
    lower_simplices = ch.simplices[ch.equations[:, 1] < -1e-12]
    hull_idx = np.unique(lower_simplices.ravel())
    hull = np.column_stack([hull_x[hull_idx], hull_y[hull_idx]])
    hull = hull[np.argsort(hull[:, 0])]

    ax.scatter(hull[:, 0], hull[:, 1], s=140, facecolors="none",
            edgecolors="black", lw=1.2, zorder=4, clip_on=False)

    hull_line, = ax.plot(
        hull[:, 0], hull[:, 1],
        color="black", lw=1.2, ls="--", zorder=1,
    )

    ax.legend(
        [(outer, inner), non_pol, hull_line],
        ["Spin polarized", "Non spin polarized", "Convex hull"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower right",
        fontsize=12,
    )

    # Add labels
    styles = [
        ("Chemical formula", 12, "black"),
        ("Space group",      8, "black"),
        ("Additional label",  6, "gray"),
    ]

    for idx, row in df.iterrows():
        parts: list[Artist] = [
            TextArea(str(row[col]), textprops=dict(fontsize=fs, color=color))
            for col, fs, color in styles
            if pd.notna(row[col])
        ]
        if not parts:
            continue

        line = HPacker(children=parts, align="baseline", pad=0, sep=2)
        rows: list[Artist] = [line]

        if pd.notna(row.get("Date")):
            rows.append(TextArea(str(row["Date"]), textprops=dict(fontsize=6, color="gray")))

        box = VPacker(children=rows, align="left", pad=0, sep=1)

        ab = AnnotationBbox(
            box, (row["Al fraction"], row["Formation energy (Ry)"]),
            xybox=(12, 8), boxcoords="offset points",
            box_alignment=(0, 0.5),
            frameon=False,
            annotation_clip=False,
        )
        ax.add_artist(ab)


    # Set limits
    ax.set_xlim(0, 25)
    ax.margins(y=0.5)
    ax.set_ylim(top=0)

    # Move x axis ticks to top
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(left=True, labelleft=True, right=False, labelright=False)

    # Labels
    ax.set_xlabel("% Al", fontsize=16)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(r"$E_{\mathrm{form}}$ (Ry/atom)", fontsize=16)
    ax.grid(alpha=0.6)

    plt.savefig("Phase Diagram.png", dpi=360)
    fig.savefig("Phase Diagram.pdf", bbox_inches="tight")   # vector, good for the report

    plt.show()

def phase_diagram_ce():
    df = pd.read_csv(DATA_PATH)
    df = df[df["Include in plot"] == True]

    dft_x = df["Al fraction"].to_numpy()          # check units: must be in % like the CE data
    dft_y = df["Formation energy (Ry)"].to_numpy()
    # Use the file column if your DFT csv has one, otherwise fall back to the index
    dft_tags = (df["File"] if "File" in df.columns else df.index.to_series()).astype(str).to_numpy()

    fractions, predicted_energies, tags = list_all_energies()

    fig, ax = plt.subplots(figsize=(16, 10))

    # DFT data (two overlaid markers for the red ring / yellow centre look)
    outer = ax.scatter(dft_x, dft_y, s=60, color=RED, zorder=3, clip_on=False)
    inner = ax.scatter(dft_x, dft_y, s=20, color=YELLOW, zorder=4, clip_on=False)

    # Cluster expansion predictions
    predicted_data = ax.scatter(
        fractions, predicted_energies,
        alpha=0.5, s=20, color="#0084FF", zorder=2, clip_on=False,
    )

    # Convex hull over DFT + predictions (pure elements have E_form = 0)
    ref_x = [0.0, 100.0]
    ref_y = [0.0, 0.0]
    hull_x = np.concatenate([dft_x, fractions, ref_x])
    hull_y = np.concatenate([dft_y, predicted_energies, ref_y])

    points = np.column_stack([hull_x / 100, hull_y])
    ch = ConvexHull(points)
    lower_simplices = ch.simplices[ch.equations[:, 1] < -1e-12]
    hull_idx = np.unique(lower_simplices.ravel())
    hull = np.column_stack([hull_x[hull_idx], hull_y[hull_idx]])
    hull = hull[np.argsort(hull[:, 0])]

    ax.scatter(hull[:, 0], hull[:, 1], s=140, facecolors="none",
               edgecolors="black", lw=1.2, zorder=4, clip_on=False)
    hull_line, = ax.plot(hull[:, 0], hull[:, 1],
                         color="black", lw=1.2, ls="--", zorder=1)

    # Hover tooltip covering both datasets
    sources = [
        (inner, dft_x, dft_y, dft_tags, "DFT calculation"),
        (predicted_data, fractions, predicted_energies, tags, "CE prediction"),
    ]
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(15, -45), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
        arrowprops=dict(arrowstyle="->"), zorder=10,
    )
    annot.set_visible(False)

    def on_hover(event):
        visible = annot.get_visible()
        if event.inaxes == ax:
            for artist, xs, ys, labels, kind in sources:
                hit, info = artist.contains(event)
                if hit:
                    i = info["ind"][0]
                    annot.xy = (xs[i], ys[i])
                    annot.set_text(
                        f"{labels[i]}\n"
                        f"[{kind}]\n"
                        f"{xs[i]:.1f}% Al\n"
                        f"{ys[i]:.5f} Ry"
                    )
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if visible:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    # Limits and axes
    ax.set_xlim(0, 25)
    ax.margins(y=0.5)
    ax.set_ylim(top=0)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(left=True, labelleft=True, right=False, labelright=False)

    # Labels
    ax.set_xlabel("Calculated and predicted energy as a function of % Al", fontsize=16)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(r"$E_{\mathrm{form}}$ (Ry/atom)", fontsize=16)
    ax.grid(alpha=0.6)

    ax.legend(
        [(outer, inner), predicted_data, hull_line],
        ["DFT data", "Cluster expansion predictions", "Convex hull"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower right",
        fontsize=12,
    )

    fig.savefig("Phase Diagram Cluster Expansion.png", dpi=360)
    fig.savefig("Phase Diagram Cluster Expansion.pdf", bbox_inches="tight")

    plt.show()



if __name__ == "__main__":
    phase_diagram()
    phase_diagram_ce()