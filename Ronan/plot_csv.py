
"""
KPOINTS convergence plotting file for Quantum ESPRESSO

Author: Des De Borger en ronana
Modified: 06/11/2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

csv_path = r"C:\Users\vande\Documents\GitHub\cmp-fe-al-phase-diagram\Ronan\Step2\Bcc Fe\results.csv"

# Dataset
stage_filter = "ECUTRHO"

# Columns
x = "setting"
y = "Column1"

# Axis labels and title
xlabel = "ecutrho cutoff"
ylabel = "Change in pressure (kbar)"
title = "ecutrho convergence test"

# CSV format
sep = ";"
decimal = ","

# Plot appearance
color = "#2471A3"
marker = "o"
linestyle = "-"
linewidth = 2.2
markersize = 7

grid = True
show_legend = True

# Reference lines
show_zero_line = False
show_tolerance = False
tolerance = 1.0  # kbar

# Figure
figsize = (8, 5)
dpi = 150

# Save plot (None = don't save)
filename = None


# ============================================================
# READ AND FILTER DATA
# ============================================================

df = pd.read_csv(
    Path(csv_path),
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

# Clean column headers (remove whitespace)
df.columns = df.columns.str.strip()

# Debug: show detected columns
print("Detected columns:", df.columns.tolist())

# Check that required columns exist
if "stage" not in df.columns:
    raise ValueError(
        "Column 'stage' not found. "
        f"Detected columns: {df.columns.tolist()}"
    )

# Select KPOINTS data
df = df[df["stage"].astype(str).str.strip() == stage_filter].copy()

# Check that data was found
if df.empty:
    raise ValueError(f"No data found for stage '{stage_filter}'")

df = df[[x, y]].dropna()

# Remove the initial reference point (zero pressure difference)
df = df[df[y] != 0]

df = df.sort_values(x)

print("\nData being plotted:")
print(df.to_string(index=False))


# ============================================================
# CREATE PLOT
# ============================================================

fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Main curve
ax.plot(
    df[x],
    df[y],
    color=color,
    marker=marker,
    linestyle=linestyle,
    linewidth=linewidth,
    markersize=markersize,
    markerfacecolor="white",
    markeredgecolor=color,
    markeredgewidth=1.8,
    label="Pressure difference",
    zorder=3
)

# Zero reference line
if show_zero_line:
    ax.axhline(
        0,
        color="#555555",
        linewidth=1.2,
        linestyle="-",
        alpha=0.8,
        label="Zero difference",
        zorder=1
    )

# Convergence tolerance
if show_tolerance:
    ax.axhline(
        tolerance,
        color="#D35400",
        linewidth=1.3,
        linestyle="--",
        alpha=0.8,
        label=f"Tolerance (±{tolerance} kbar)",
        zorder=1
    )

    ax.axhline(
        -tolerance,
        color="#D35400",
        linewidth=1.3,
        linestyle="--",
        alpha=0.8,
        zorder=1
    )

# Axis labels
ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
ax.set_ylabel(ylabel, fontsize=12, labelpad=10)

# Title
ax.set_title(
    title,
    fontsize=15,
    fontweight="semibold",
    pad=16
)

# Grid
if grid:
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.3
    )

# Axis styling
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#444444")
ax.spines["bottom"].set_color("#444444")

ax.tick_params(
    axis="both",
    labelsize=10,
    direction="out",
    length=4,
    colors="#444444"
)

# Integer x-axis ticks
ax.set_xticks(df[x])

# Legend
if show_legend:
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="best"
    )

fig.tight_layout()

# Save
if filename:
    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

# Display
plt.show()
