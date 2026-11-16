import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
from matplotlib.legend_handler import HandlerTuple
from scipy.spatial import ConvexHull

DATA_PATH = "phase_diagram_data.csv"

RED = "#cc0101"
YELLOW = "#f1ed00"

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