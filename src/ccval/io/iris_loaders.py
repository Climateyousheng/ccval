from __future__ import annotations

import glob
import os
import iris

from .analysis import try_extract


def load_annual_mean_cubes(expt: str, base_dir: str = "~/annual_mean") -> iris.cube.CubeList:
    base_dir = os.path.expanduser(base_dir)
    root = os.path.join(base_dir, expt)
    filenames = glob.glob(os.path.join(root, "**/*.nc"), recursive=True)
    return iris.load(filenames)


def extract_soilparam_cubes(cubes: iris.cube.CubeList):
    """
    Returns dict of CubeLists (as returned by try_extract).
    Keys: rh, cs, cv, frac, gpp, npp, fgco2, tas, pr
    """
    frac = try_extract(cubes, "frac")
    if not frac:
        frac = try_extract(cubes, 3317)

    return {
        "rh": try_extract(cubes, "rh"),
        "cs": try_extract(cubes, "cs"),
        "cv": try_extract(cubes, "cv"),
        "frac": frac,
        "gpp": try_extract(cubes, "gpp"),
        "npp": try_extract(cubes, "npp"),
        "fgco2": try_extract(cubes, "fgco2"),
        "tas": try_extract(cubes, "tas"),
        "pr": try_extract(cubes, "pr"),
    }


def first_cube(cubelist_or_cube):
    if cubelist_or_cube is None:
        return None
    if isinstance(cubelist_or_cube, iris.cube.Cube):
        return cubelist_or_cube
    return cubelist_or_cube[0] if len(cubelist_or_cube) else None