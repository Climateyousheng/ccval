from __future__ import annotations

import glob
import os
import re
import iris


def load_annual_mean_cubes(expt: str, base_dir: str = "~/annual_mean") -> iris.cube.CubeList:
    base_dir = os.path.expanduser(base_dir)
    root = os.path.join(base_dir, expt)
    filenames = glob.glob(os.path.join(root, "**/*.nc"), recursive=True)
    return iris.load(filenames)


MONTH_MAP_ALPHA = {
    "ja": 1,
    "fb": 2,
    "mr": 3,
    "ar": 4,
    "my": 5,
    "jn": 6,
    "jl": 7,
    "ag": 8,
    "sp": 9,
    "ot": 10,
    "nv": 11,
    "dc": 12,
}


def decode_month(mon_code: str) -> int:
    if not mon_code:
        return 0
    s = mon_code.lower()

    if s.isalpha():
        return MONTH_MAP_ALPHA.get(s, 0)

    if len(s) == 2:
        first = s[0]
        if first.isdigit():
            m = int(first)
            return m if 1 <= m <= 9 else 0
        if first in ("a", "b", "c"):
            return {"a": 10, "b": 11, "c": 12}[first]

    return 0


def find_matching_files(
    expt_name: str,
    model: str,
    up: str,
    start_year: int | None = None,
    end_year: int | None = None,
    base_dir: str = "~/dump2hold",
):
    """
    Find matching UM output files for a given experiment and sort them by year/month.

    Supports:
      - xqhuja#pi000001853dc+
      - xqhujo#da00000185511+  (11=Jan, 91=Sep, a1=Oct, b1=Nov, c1=Dec)
    """
    base_dir = os.path.expanduser(base_dir)
    datam_path = os.path.join(base_dir, expt_name, "datam")
    if not os.path.isdir(datam_path):
        datam_path = base_dir

    pattern = (
        fr"{re.escape(expt_name)}[{model}]\#{re.escape(up)}00000"
        fr"(\d{{4}})"
        fr"([a-zA-Z]{{2}}|[0-9a-cA-C][0-9])"
        fr"\+"
    )
    regex = re.compile(pattern)

    files = glob.glob(os.path.join(datam_path, "**"), recursive=True)
    matching_files = []

    for f in files:
        match = regex.search(os.path.basename(f)) or regex.search(f)
        if not match:
            continue

        year = int(match.group(1))
        month = decode_month(match.group(2))
        if month == 0:
            continue

        if (start_year is None or year >= start_year) and (end_year is None or year <= end_year):
            matching_files.append((year, month, f))

    matching_files.sort(key=lambda x: (x[0], x[1]))
    return matching_files


def first_cube(cubelist_or_cube):
    if cubelist_or_cube is None:
        return None
    if isinstance(cubelist_or_cube, iris.cube.Cube):
        return cubelist_or_cube
    return cubelist_or_cube[0] if len(cubelist_or_cube) else None