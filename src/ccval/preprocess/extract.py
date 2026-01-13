from __future__ import annotations

import iris
from iris import Constraint
import numpy as np


def _msi_from_stash_obj(st):
    if st is None:
        return None
    try:
        return f"m{int(st.model):02d}s{int(st.section):02d}i{int(st.item):03d}"
    except Exception:
        pass
    try:
        s = str(st.msi).strip()
        if s.startswith("m") and "s" in s and "i" in s:
            return s
    except Exception:
        pass
    return None


def _msi_from_numeric_stash_code(code):
    if code is None:
        return None
    try:
        n = int(code)
    except Exception:
        return None
    section = n // 1000
    item = n % 1000
    model = 2 if section >= 30 else 1
    return f"m{model:02d}s{section:02d}i{item:03d}"


def _msi_from_any_attr(attrs):
    if not attrs:
        return None
    if "STASH" in attrs:
        msi = _msi_from_stash_obj(attrs.get("STASH"))
        if msi:
            return msi
    if "stash_code" in attrs:
        msi = _msi_from_numeric_stash_code(attrs.get("stash_code"))
        if msi:
            return msi
    return None


def try_extract(cubes, code, stash_lookup_func=None, debug: bool = False):
    candidates = [code]

    if stash_lookup_func is not None and isinstance(code, str):
        msi = stash_lookup_func(code)
        if msi and msi != "nothing":
            candidates.append(msi)

    try:
        candidates.append(str(code))
    except Exception:
        pass

    if isinstance(code, (int, np.integer)) or (isinstance(code, str) and code.isdigit()):
        try:
            candidates.append(int(code))
        except Exception:
            pass

    cand_msi = set()
    for c in candidates:
        if isinstance(c, str) and c.startswith("m") and "s" in c and "i" in c:
            cand_msi.add(c.strip())
            continue
        msi = _msi_from_numeric_stash_code(c)
        if msi:
            cand_msi.add(msi)

    if debug:
        print(f"Trying to extract cube for candidates: {candidates}")
        print(f"Normalized candidate MSIs: {cand_msi}")

    def _match(c):
        attrs = getattr(c, "attributes", {}) or {}
        cube_msi = _msi_from_any_attr(attrs)
        if debug:
            print(f"Cube: {c.name()} attrs keys={list(attrs.keys())} -> MSI={cube_msi}")
        return cube_msi in cand_msi

    try:
        return cubes.extract(Constraint(cube_func=_match))
    except Exception:
        return iris.cube.CubeList([])


def first_cube(cubelist_or_cube):
    if cubelist_or_cube is None:
        return None
    if isinstance(cubelist_or_cube, iris.cube.Cube):
        return cubelist_or_cube
    return cubelist_or_cube[0] if len(cubelist_or_cube) else None


def extract_soilparam_cubes(cubes: iris.cube.CubeList):
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
        # currently only include variables in pi/pt files, but can be extended later
        # "fgco2": try_extract(cubes, "fgco2"),
        # "tas": try_extract(cubes, "tas"),
        # "pr": try_extract(cubes, "pr"),
    }