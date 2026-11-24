import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
from matplotlib.legend_handler import HandlerTuple
from scipy.spatial import ConvexHull

from Des.cluster_expansion import list_all_energies, find_cif
from Des.utils import get_equilibrium_volumes

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
    full_df = pd.read_csv(DATA_PATH)
    df = full_df[full_df["Include in plot"] == True]

    dft_x = df["Al fraction"].to_numpy()
    dft_y = df["Formation energy (Ry)"].to_numpy()
    dft_tags = df["Index"].astype(str).to_numpy()

    fractions, predicted_energies, tags = list_all_energies()
    fractions = np.asarray(fractions)
    predicted_energies = np.asarray(predicted_energies)
    tags = np.asarray(tags)

    # Filenames of every structure that has a DFT calculation
    calculated_files = set()
    for idx, formula in zip(full_df["Index"], full_df["Chemical formula"]):
        try:
            calculated_files.add(find_cif(int(idx), formula).name)
        except (FileNotFoundError, ValueError):
            pass  # no cif found, or ambiguous match
    done = np.isin(tags, list(calculated_files))
    print(f"{done.sum()} of {len(tags)} predictions already have a DFT calculation")

    fig, ax = plt.subplots(figsize=(16, 10))

    # DFT data
    outer = ax.scatter(dft_x, dft_y, s=60, color=RED, zorder=3, clip_on=False)
    inner = ax.scatter(dft_x, dft_y, s=20, color=YELLOW, zorder=4, clip_on=False)

    # Predictions not yet calculated
    pending = ax.scatter(
        fractions[~done], predicted_energies[~done],
        alpha=0.5, s=20, color="#0084FF", zorder=2, clip_on=False,
    )
    # Predictions that have already been calculated with DFT
    calculated = ax.scatter(
        fractions[done], predicted_energies[done],
        s=45, facecolors="none", edgecolors="#00A650",
        lw=1.5, zorder=3, clip_on=False,
    )

        # Convex hull over DFT data only (pure elements have E_form = 0)
    ref_x = [0.0, 100.0]
    ref_y = [0.0, 0.0]
    hull_x = np.concatenate([dft_x, ref_x])
    hull_y = np.concatenate([dft_y, ref_y])

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

    # Predictions that fall below the DFT hull are candidates worth calculating
    hull_at_pred = np.interp(fractions, hull[:, 0], hull[:, 1])
    below = (predicted_energies < hull_at_pred - 1e-6) & ~done
    print(f"{below.sum()} uncalculated predictions lie below the DFT hull:")
    for t, x, y in zip(tags[below], fractions[below], predicted_energies[below]):
        print(f"  {t}  {x:.1f}% Al  {y:.5f} Ry")

    # Hover tooltip covering all datasets
    sources = [
        (inner, dft_x, dft_y, dft_tags, "DFT calculation"),
        (calculated, fractions[done], predicted_energies[done], tags[done],
         "CE prediction (already calculated)"),
        (pending, fractions[~done], predicted_energies[~done], tags[~done],
         "CE prediction (not calculated)"),
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

    ax.set_xlabel("Calculated and predicted energy as a function of % Al", fontsize=16)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(r"$E_{\mathrm{form}}$ (Ry/atom)", fontsize=16)
    ax.grid(alpha=0.6)

    ax.legend(
        [(outer, inner), pending, calculated, hull_line],
        ["DFT data", "CE prediction (not yet calculated)",
         "CE prediction (already calculated)", "Convex hull (DFT)"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower right",
        fontsize=12,
    )

    fig.savefig("Phase Diagram Cluster Expansion.png", dpi=360)
    fig.savefig("Phase Diagram Cluster Expansion.pdf", bbox_inches="tight")

    plt.show()

def vegard_law():
    df = pd.read_csv(DATA_PATH)
    df = df[df["Include in plot"]==True]
    df = df[df["Volume (Ang **3)"].notna()]

    def count_atoms(formula: str) -> dict:
        """'Fe52Al2' -> {'Fe': 52, 'Al': 2}; a missing number means 1 (e.g. 'Fe63Al')."""
        counts = {}
        for element, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
            counts[element] = counts.get(element, 0) + (int(n) if n else 1)
        return counts
    
    counts = df["Chemical formula"].apply(count_atoms)
    df["n_Fe"] = counts.apply(lambda c: c.get("Fe", 0))
    df["n_Al"] = counts.apply(lambda c: c.get("Al", 0))

    df["Volume per atom (Ang**3)"] = df["Volume (Ang **3)"] / (df["n_Fe"] + df["n_Al"])

    pure_volumes = get_equilibrium_volumes()
    df["Predicted volume per atom (Ang**3)"] = df["n_Fe"] * pure_volumes["Fe"] + df["n_Al"] * pure_volumes["Al"]

    # Vegard's law: linear interpolation between the pure-element volumes per atom
    x = np.linspace(0, 25, 100)
    v_vegard = (1 - x / 100) * pure_volumes["Fe"] + (x / 100) * pure_volumes["Al"]

    fig, ax = plt.subplots(figsize=(16, 10))

    vegard_line, = ax.plot(x, v_vegard, color=BLUE, lw=2, zorder=1)

    outer = ax.scatter(df["Al fraction"], df["Volume per atom (Ang**3)"],
                       s=60, color=RED, zorder=2, clip_on=False)
    inner = ax.scatter(df["Al fraction"], df["Volume per atom (Ang**3)"],
                       s=20, color=YELLOW, zorder=3, clip_on=False)

    ax.set_xlim(0, 25)
    ax.set_ylim(10, 13)
    ax.margins(y=0.1)
    ax.tick_params(labelsize=12)
    ax.set_xlabel("% Al", fontsize=16)
    ax.set_ylabel(r"Volume per atom ($\mathrm{\AA}^3$)", fontsize=16)
    ax.set_title("Calculated and predicted volume as a function of % Al", fontsize=16)
    ax.grid(alpha=0.6)

    ax.legend(
        [(outer, inner), vegard_line],
        ["DFT", "Vegard's law"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper left",
        fontsize=12,
    )

    fig.savefig("Vegard Law.png", dpi=360, bbox_inches="tight")
    fig.savefig("Vegard Law.pdf", bbox_inches="tight")
    plt.show()
if __name__ == "__main__":
    vegard_law()
    phase_diagram()