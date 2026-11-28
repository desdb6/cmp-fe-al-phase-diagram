import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import linregress
from scipy.spatial import ConvexHull

from Des.cluster_expansion import list_all_energies, find_cif
from Des.utils import get_equilibrium_volumes

DATA_PATH = "phase_diagram_data.csv"

RED = "#cc0101"
YELLOW = "#f1ed00"
BLUE = "#0084FF"
GRAY = "#4D4D4D"

RY_TO_MEV = 13_605.693

def place_labels(fig, ax, df, make_box, hull, legend=None,
                 obstacle_df=None, point_pad=5, line_pad=3,
                 box_pad=3, gap=6, max_vertical_steps=8):
    """Place labels next to points, preferring the right side, without overlaps."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    k = fig.dpi / 72.0

    df = df.reset_index(drop=True)
    X = df["Al fraction"].to_numpy(float)
    Y = df["Formation energy (meV)"].to_numpy(float)
    pts = ax.transData.transform(np.column_stack([X, Y]))

    # All plotted data points are obstacles, including unlabelled ones
    if obstacle_df is None:
        obstacle_df = df

    marker_pts = ax.transData.transform(
        np.column_stack([
            obstacle_df["Al fraction"].to_numpy(float),
            obstacle_df["Formation energy (meV)"].to_numpy(float)
        ])
    )

    # Hull line is also an obstacle
    seg = [
        np.column_stack([
            np.linspace(a[0], b[0], 80),
            np.linspace(a[1], b[1], 80)
        ])
        for a, b in zip(hull[:-1], hull[1:])
    ]

    hull_px = ax.transData.transform(np.vstack(seg))
    fixed = np.vstack([marker_pts, hull_px])

    fixed_pad = np.concatenate([
        np.full(len(marker_pts), point_pad),
        np.full(len(hull_px), line_pad)
    ])

    # Keep track of already occupied label positions
    boxes = np.empty((0, 4))

    if legend is not None:
        bb = legend.get_window_extent(renderer)
        boxes = np.vstack([boxes, [bb.x0, bb.y0, bb.x1, bb.y1]])

    xmin, xmax = ax.bbox.x0 + 4, fig.bbox.x1 - 4
    ymin, ymax = ax.bbox.y0 + 4, ax.bbox.y1 - 4

    def measure(row):
        tmp = AnnotationBbox(
            make_box(row), (0, 0), xybox=(0, 0),
            boxcoords="offset points",
            box_alignment=(0.5, 0.5),
            pad=0, frameon=False
        )
        ax.add_artist(tmp)
        bb = tmp.get_window_extent(renderer)
        tmp.remove()
        return bb.width, bb.height

    def count_hits(x0, y0, x1, y1):
        hits = 0

        # Previously placed labels
        if len(boxes):
            hits += np.count_nonzero(
                (x0 < boxes[:, 2] + box_pad) &
                (x1 > boxes[:, 0] - box_pad) &
                (y0 < boxes[:, 3] + box_pad) &
                (y1 > boxes[:, 1] - box_pad)
            )

        # Data points and hull line
        hits += np.count_nonzero(
            (fixed[:, 0] > x0 - fixed_pad) &
            (fixed[:, 0] < x1 + fixed_pad) &
            (fixed[:, 1] > y0 - fixed_pad) &
            (fixed[:, 1] < y1 + fixed_pad)
        )

        return hits

    # Prioritise hull points, then points closest to the hull
    is_hull = np.array([
        np.any(
            np.isclose(x, hull[:, 0], atol=1e-9) &
            np.isclose(y, hull[:, 1], atol=1e-12)
        )
        for x, y in zip(X, Y)
    ])

    above_hull = np.maximum(
        0,
        Y - np.interp(X, hull[:, 0], hull[:, 1])
    )

    crowd = (
        np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2) < 40
    ).sum(axis=1)

    order = np.lexsort((-crowd, above_hull, ~is_hull))

    for i in order:
        row = df.iloc[i]

        if make_box(row) is None:
            continue

        w, h = measure(row)
        px, py = pts[i]

        # Vertical stacking positions, starting directly beside the dot
        step = max(8.0, h / k + 2.0)
        vertical_offsets = [0.0]

        for n in range(1, max_vertical_steps + 1):
            vertical_offsets.extend([n * step, -n * step])

        best = None

        # Try the right side first; only use the left if necessary
        for side in ("right", "left"):
            for dy in vertical_offsets:

                if side == "right":
                    cx = px + gap * k + w / 2
                else:
                    cx = px - gap * k - w / 2

                cy = py + dy * k

                x0, x1 = cx - w / 2, cx + w / 2
                y0, y1 = cy - h / 2, cy + h / 2

                # Stay within the available figure space
                if x0 < xmin or x1 > xmax or y0 < ymin or y1 > ymax:
                    continue

                if count_hits(x0, y0, x1, y1) == 0:
                    best = (cx, cy, x0, y0, x1, y1)
                    break

            if best is not None:
                break

        if best is None:
            print(f"Could not place label for row {i}; skipped")
            continue

        cx, cy, x0, y0, x1, y1 = best

        # No arrowprops: labels have no connecting lines
        ab = AnnotationBbox(
            make_box(row),
            (X[i], Y[i]),
            xybox=((cx - px) / k, (cy - py) / k),
            xycoords="data",
            boxcoords="offset points",
            box_alignment=(0.5, 0.5),
            pad=0,
            frameon=False,
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-",
                color="gray",
                lw=0.6,
                shrinkA=0,
                shrinkB=3,
                connectionstyle="arc3",
            ),
        )

        ax.add_artist(ab)
        boxes = np.vstack([boxes, [x0, y0, x1, y1]])

def phase_diagram(labels="near hull and lowest", show_extra=False, near_hull_mev=4.0):
    """labels: 'all' or 'hull' (only hull vertices).
    show_extra: also show the 'Additional label' and the date under each label."""
    df = pd.read_csv(DATA_PATH)
    df = df[df["Include in plot"] == True].reset_index(drop=True)
    df["Formation energy (meV)"] = df["Formation energy (Ry)"] * RY_TO_MEV

    spin_polarized_df = df[df["Spin polarized"] == True]

    fig, ax = plt.subplots(figsize=(14, 5.5))

    outer = ax.scatter(
        x=spin_polarized_df["Al fraction"],
        y=spin_polarized_df["Formation energy (meV)"],
        s=40, color=RED, zorder=2, clip_on=False
    )
    inner = ax.scatter(
        x=spin_polarized_df["Al fraction"],
        y=spin_polarized_df["Formation energy (meV)"],
        s=10, color=YELLOW, zorder=3, clip_on=False
    )

    # Convex hull (pure elements have E_form = 0 by definition)
    ref_x = [0.0, 100.0]
    ref_y = [0.0, 0.0]
    hull_x = np.concatenate([df["Al fraction"].to_numpy(), ref_x])
    hull_y = np.concatenate([df["Formation energy (meV)"].to_numpy(), ref_y])   # df, not spin_polarized_df
    points = np.column_stack([hull_x / 100, hull_y])
    ch = ConvexHull(points)
    lower_simplices = ch.simplices[ch.equations[:, 1] < -1e-12]
    hull_idx = np.unique(lower_simplices.ravel())
    hull = np.column_stack([hull_x[hull_idx], hull_y[hull_idx]])
    hull = hull[np.argsort(hull[:, 0])]

    # Which rows of df are hull vertices (computed once, reused below)
    hull_mask = np.array([
        np.any(np.isclose(x, hull[:, 0], atol=1e-9) &
               np.isclose(y, hull[:, 1], atol=1e-6))
        for x, y in zip(df["Al fraction"], df["Formation energy (meV)"])
    ])

    # Print the hull structures
    hull_df = df[hull_mask].sort_values("Al fraction")
    print("\nConvex hull structures (meV/atom):")
    print(hull_df[["Chemical formula", "Space group", "Al fraction",
                   "Formation energy (meV)"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    ax.scatter(hull[:, 0], hull[:, 1], s=140, facecolors="none",
            edgecolors="black", lw=1.2, zorder=4, clip_on=False)
    hull_line, = ax.plot(hull[:, 0], hull[:, 1], color="black", lw=1.2, ls="--", zorder=1)

    leg = ax.legend(
        [(outer, inner), hull_line],
        ["DFT calculation", "Convex hull"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower left",
        fontsize=12,
    )

    # Limits, axes (must be final before placing labels)
    ax.set_xlim(0, 25)
    ax.margins(y=0.1)
    ax.set_ylim(top=0)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(left=True, labelleft=True, right=False, labelright=False)
    ax.set_xlabel("% Al", fontsize=12)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(r"$E_{\mathrm{form}}$ (meV/atom)", fontsize=12)
    ax.grid(alpha=0.6)
    fig.subplots_adjust(left=0.07, right=0.80, top=0.90, bottom=0.06)  # right margin holds labels

    # Labels
    styles = [("Chemical formula", 9, "black"), ("Space group", 7, "black")]
    if show_extra:
        styles.append(("Additional label", 6, "gray"))

    def make_box(row):
        parts = [TextArea(str(row[col]), textprops=dict(fontsize=fs, color=color))
                 for col, fs, color in styles if pd.notna(row[col])]
        if not parts:
            return None
        rows = [HPacker(children=parts, align="baseline", pad=0, sep=2)]
        if show_extra and pd.notna(row.get("Date")):
            rows.append(TextArea(str(row["Date"]), textprops=dict(fontsize=6, color="gray")))
        return VPacker(children=rows, align="left", pad=0, sep=1)

    label_df = df

    if labels == "hull":
        on_hull = [
            np.any(
                np.isclose(x, hull[:, 0], atol=1e-9) &
                np.isclose(y, hull[:, 1], atol=1e-12)
            )
            for x, y in zip(df["Al fraction"], spin_polarized_df["Formation energy (meV)"])
        ]
        label_df = df[on_hull]

    elif labels == "near_hull":
        hull_at_points = np.interp(
            df["Al fraction"],
            hull[:, 0],
            hull[:, 1]
        )

        energy_above_hull_mev = (
            spin_polarized_df["Formation energy (meV)"].to_numpy() - hull_at_points
        )

        label_df = df[energy_above_hull_mev <= near_hull_mev]

    elif labels == "near hull and lowest":
        # points on the convex hull
        on_hull = np.array([
            np.any(
                np.isclose(x, hull[:, 0], atol=1e-9) &
                np.isclose(y, hull[:, 1], atol=1e-12)
            )
            for x, y in zip(df["Al fraction"], spin_polarized_df["Formation energy (meV)"])
        ])

        # lowest-energy point for every chemical formula
        # (covers different cell sizes at the same Al fraction)
        lowest_idx = df.groupby("Chemical formula")["Formation energy (meV)"].idxmin()
        lowest = df.index.isin(lowest_idx)

        # always label the reference structures
        is_ref = df["Space group"].astype(str).str.contains(r"D03|L1_2", na=False).to_numpy()

        label_df = df[on_hull | lowest | is_ref]

    elif labels != "all":
        raise ValueError("labels must be 'all', 'hull', or 'near_hull'")

    place_labels(
        fig, ax, label_df, make_box, hull,
        legend=leg,
        obstacle_df=df
    )

    plt.savefig("Phase Diagram.png", dpi=360)
    fig.savefig("Phase Diagram.pdf", bbox_inches="tight")
    plt.show()

def phase_diagram_ce():
    full_df = pd.read_csv(DATA_PATH)
    df = full_df[full_df["Include in plot"] == True]

    dft_x = df["Al fraction"].to_numpy()
    dft_y = df["Formation energy (Ry)"].to_numpy() * RY_TO_MEV
    dft_tags = df["Index"].astype(str).to_numpy()

    fractions, predicted_energies, tags = list_all_energies()
    fractions = np.asarray(fractions)
    predicted_energies = np.asarray(predicted_energies) * RY_TO_MEV
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

    fig, ax = plt.subplots(figsize=(7, 5))

    # DFT data
    outer = ax.scatter(dft_x, dft_y, s=40, color=RED, zorder=3, clip_on=False)
    inner = ax.scatter(dft_x, dft_y, s=10, color=YELLOW, zorder=4, clip_on=False)

    # Predictions not yet calculated
    pending = ax.scatter(
        fractions[~done], predicted_energies[~done],
        alpha=0.3, s=20, color="#00BB5A", zorder=2, clip_on=False,
    )
    # Predictions that have already been calculated with DFT0084FF
    calculated = ax.scatter(
        fractions[done], predicted_energies[done],
        s=60, facecolors="none", edgecolors="#006CD1", alpha=0.8,
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
    ax.margins(y=0.2)
    ax.set_ylim(top=0)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(left=True, labelleft=True, right=False, labelright=False)

    ax.set_xlabel("Calculated and predicted energy as a function of % Al", fontsize=12)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(r"$E_{\mathrm{form}}$ (meV/atom)", fontsize=16)
    ax.grid(alpha=0.6)

    ax.legend(
        [(outer, inner), pending, calculated, hull_line],
        ["DFT data", "CE prediction",
         "CE prediction (validated with DFT)", "Convex hull (DFT)"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower left",
        fontsize=8,
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
    df.loc[df["Chemical formula"] == "bcc-Fe", "n_Fe"] = 2

    df["Volume per atom (Ang**3)"] = df["Volume (Ang **3)"] / (df["n_Fe"] + df["n_Al"])

    pure_volumes = get_equilibrium_volumes()
    df["Predicted volume per atom (Ang**3)"] = df["n_Fe"] * pure_volumes["Fe"] + df["n_Al"] * pure_volumes["Al"]

    x = np.linspace(0, 25, 100)
    # Vegard's law: V = intercept + slope * x   (x in % Al)
    vegard_slope = (pure_volumes["Al"] - pure_volumes["Fe"]) / 100
    vegard_intercept = pure_volumes["Fe"]
    v_vegard = vegard_slope * x + vegard_intercept

    # Linear fit
    linreg_result = linregress(df["Al fraction"], df["Volume per atom (Ang**3)"])
    v_fit = linreg_result.slope * x + linreg_result.intercept

    n_points = len(df)
    dof = n_points - 2

    print("=" * 55)
    print("Linear fit: V = intercept + slope * x   (x in % Al)")
    print("=" * 55)
    print(f"Number of points : {n_points}")
    print(f"Intercept        : {linreg_result.intercept:.4f} ± {linreg_result.intercept_stderr:.4f} Å³/atom")
    print(f"Slope            : {linreg_result.slope:.5f} ± {linreg_result.stderr:.5f} Å³/atom per % Al")
    print()
    print("Vegard's law")
    print(f"Intercept (V_Fe) : {vegard_intercept:.4f} Å³/atom")
    print(f"Slope            : {vegard_slope:.5f} Å³/atom per % Al")
    print(f"Fit - Vegard slope: {linreg_result.slope - vegard_slope:+.5f} "
        f"({(linreg_result.slope - vegard_slope) / linreg_result.stderr:+.1f} σ)")
    print("=" * 55)

    vegard_label = (rf"Vegard's law: $V = ({vegard_intercept:.3f} {vegard_slope:+.4f}\,\mathrm{{\%Al}})\ \mathrm{{\AA}}^3$")
    fit_label = (rf"Linear fit: $V = ({linreg_result.intercept:.3f} {linreg_result.slope:+.4f}\,\mathrm{{\%Al}})\ \mathrm{{\AA}}^3$")

    fig, ax = plt.subplots(figsize=(7, 5))

    vegard_line, = ax.plot(x, v_vegard, color=BLUE, lw=2, zorder=1)
    fit_line, = ax.plot(x, v_fit, color=GRAY, linestyle="--", alpha=0.7, lw=2, zorder=1)

    outer = ax.scatter(df["Al fraction"], df["Volume per atom (Ang**3)"],
                       s=40, color=RED, zorder=2, clip_on=False)
    inner = ax.scatter(df["Al fraction"], df["Volume per atom (Ang**3)"],
                       s=10, color=YELLOW, zorder=3, clip_on=False)

    ax.set_xlim(0, 25)
    ax.set_ylim(11.25, 12.75)
    ax.margins(y=0.1)
    ax.tick_params(labelsize=12)
    ax.set_xlabel("% Al", fontsize=12)
    ax.set_ylabel(r"Volume per atom ($\mathrm{\AA}^3$)", fontsize=12)
    ax.set_title("Calculated and predicted volume as a function of % Al", fontsize=12)
    ax.grid(alpha=0.6)

    ax.legend(
        [(outer, inner), vegard_line, fit_line],
        ["DFT", vegard_label, fit_label],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper left",
        fontsize=8,
    )

    fig.savefig("Vegard Law.png", dpi=360, bbox_inches="tight")
    fig.savefig("Vegard Law.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    vegard_law()
    phase_diagram()
    phase_diagram_ce()