# NOTE: Imported from user prototype 'soil_params_figures.py'. Contains placeholders ('...') to fill.
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import cartopy.crs as ccrs


def plot_diff_map(
    diff_cube,
    title: str,
    outpath: str,
    vmin: float,
    vmax: float,
    show: bool = True,
):
    """
    Plot a single 2D difference map on a PlateCarree projection and save it.

    Parameters
    ----------
    diff_cube
        2D Iris cube (lat/lon) to plot.
    title
        Title string for the plot.
    outpath
        Output image path. Parent directories are created if needed.
    vmin, vmax
        Plot range; uses 21 levels between vmin and vmax and `bwr` colormap.
    show
        If True, display interactively; if False, only save to disk (recommended in loops).

    Example
    -------
    >>> plot_diff_map(
    ...   diff_cube=diffs["tas_diff"][0,:,:],
    ...   title="SAT diff (xqhuc - xqhsh)",
    ...   outpath=os.path.expanduser("~/plots/soil_params/tas_diff.png"),
    ...   vmin=-5, vmax=5,
    ...   show=False,
    ... )
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig = plt.figure(figsize=(4, 3.5), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    qplt.contourf(diff_cube, levels=np.linspace(vmin, vmax, 21), cmap="bwr", extend="both")
    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)