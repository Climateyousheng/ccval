# NOTE: Imported from user prototype 'soil_params_maps.py'. Contains placeholders ('...') to fill.
from __future__ import annotations

import iris
from iris import Constraint


def time_mean_first_n(cube: iris.cube.Cube, n: int = 10) -> iris.cube.Cube:
    """
    Return the mean over the first `n` timesteps of a cube.

    Parameters
    ----------
    cube
        Iris cube with a leading time dimension.
    n
        Number of initial timesteps to include.

    Returns
    -------
    iris.cube.Cube
        A 2D (lat/lon) cube representing the mean over the first `n` timesteps.

    Example
    -------
    >>> tas_mean_10y = time_mean_first_n(tas_cube, n=10)
    """
    return cube[:n].copy().collapsed("time", iris.analysis.MEAN)


def pft_time_mean_first_n(
    frac_cube: iris.cube.Cube, pft_index_1based: int, n: int = 10
) -> iris.cube.Cube:
    """
    Extract a single PFT slice from a TRIFFID fraction cube and average the first `n` timesteps.

    Assumes a coordinate named `generic` indexes PFTs as 1..9.

    Parameters
    ----------
    frac_cube
        Fraction cube with dimensions like [time, generic, lat, lon].
    pft_index_1based
        PFT index using 1-based numbering (e.g., 1=BL, 2=NL, etc., depending on your convention).
    n
        Number of initial timesteps to include.

    Returns
    -------
    iris.cube.Cube
        2D mean fraction map for the selected PFT.

    Example
    -------
    >>> bl_mean_10y = pft_time_mean_first_n(frac_cube, pft_index_1based=1, n=10)
    """
    frac_pft = frac_cube.extract(Constraint(coord_values={"generic": pft_index_1based}))
    return time_mean_first_n(frac_pft, n=n)


def compute_pft_temp_precip_diffs(
    frac_a: iris.cube.Cube,
    frac_b: iris.cube.Cube,
    tas_a: iris.cube.Cube,
    tas_b: iris.cube.Cube,
    pr_a: iris.cube.Cube,
    pr_b: iris.cube.Cube,
    n_years: int = 10,
    pfts=(1, 2, 3, 4, 5),
):
    """
    Compute mean spatial differences (B - A) over the first `n_years` timesteps for:
      - selected PFT fractions
      - surface air temperature
      - precipitation (converted to mm/day)

    This matches the notebook’s “first 10 years mean difference” logic.

    Parameters
    ----------
    frac_a, frac_b
        TRIFFID PFT fraction cubes for experiments A and B.
    tas_a, tas_b
        Near-surface air temperature cubes for experiments A and B.
    pr_a, pr_b
        Precipitation rate cubes for experiments A and B (typically kg/m2/s or m/s equivalent).
    n_years
        Number of initial timesteps to include (your notebook uses 10).
    pfts
        Iterable of 1-based PFT indices to compute diffs for.

    Returns
    -------
    dict
        {
          "pft_diffs": {pft_index: 2D cube},
          "tas_diff": 2D cube,
          "pr_diff": 2D cube (mm/day)
        }

    Example
    -------
    >>> diffs = compute_pft_temp_precip_diffs(
    ...     frac_a=frac_xqhsh, frac_b=frac_xqhuc,
    ...     tas_a=tas_xqhsh, tas_b=tas_xqhuc,
    ...     pr_a=pr_xqhsh, pr_b=pr_xqhuc,
    ...     n_years=10, pfts=(1,2,3,4,5),
    ... )
    >>> bl_diff = diffs["pft_diffs"][1]
    >>> tas_diff = diffs["tas_diff"]
    """
    pft_diffs = {}
    for p in pfts:
        da = pft_time_mean_first_n(frac_a, p, n=n_years)
        db = pft_time_mean_first_n(frac_b, p, n=n_years)
        pft_diffs[p] = (db - da)

    tas_diff = time_mean_first_n(tas_b, n=n_years) - time_mean_first_n(tas_a, n=n_years)

    pr_diff = time_mean_first_n(pr_b, n=n_years) - time_mean_first_n(pr_a, n=n_years)
    pr_diff = pr_diff * 86400.0
    pr_diff.units = "mm/day"

    return {"pft_diffs": pft_diffs, "tas_diff": tas_diff, "pr_diff": pr_diff}