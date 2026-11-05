"""
Plotting file for quantum espresso scripting

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 21/10/2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_csv(
    csv_path,
    x,
    y,
    xlabel=None,
    ylabel=None,
    title=None,
    filename=None,
    label=None,
    sep=",",
    marker="o",
    linestyle="-",
    grid=True,
    ax=None,
    **plot_kwargs,
    ):
    """Plot one column of a CSV file against another.

    csv_path   : path to the file
    x, y       : header names of the x and y columns (y may be a list of headers)
    xlabel     : x-axis label (defaults to the header name)
    ylabel     : y-axis label (defaults to the header name)
    title      : optional title
    filename   : if given, save the figure here
    label      : legend label (defaults to the y header)
    sep        : column separator ("," for CSV, "\\t" for tab-separated files)
    ax         : existing Axes to draw on, e.g. to overlay several files
    plot_kwargs: passed on to ax.plot (color, lw, ms, ...)
    """
    df = pd.read_csv(Path(csv_path), sep=sep)
    ys = [y] if isinstance(y, str) else list(y)

    missing = [c for c in [x, *ys] if c not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found. Available: {list(df.columns)}")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for col in ys:
        d = df[[x, col]].dropna().sort_values(x)
        lbl = label if (label is not None and len(ys) == 1) else col
        ax.plot(d[x], d[col], marker=marker, linestyle=linestyle,
                label=lbl, **plot_kwargs)

    ax.set_xlabel(xlabel if xlabel is not None else x)
    ax.set_ylabel(ylabel if ylabel is not None else (ys[0] if len(ys) == 1 else ""))
    if title:
        ax.set_title(title, fontsize=8)
    if grid:
        ax.grid(alpha=0.5)
    ax.legend()

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    return fig, ax