from __future__ import annotations

# NOTE: Imported from user prototype 'analysis.py'. Refactor progressively.
import iris
from iris import Constraint
import iris.analysis.cartography as cart
from iris.analysis.cartography import area_weights
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import os
import glob
import yaml
from pathlib import Path
import re
import iris.quickplot as qplt
import xarray as xr
import cf_units
import warnings
warnings.filterwarnings("ignore", module='iris')

# Helper

def stash(s):
    switcher = {
        'tas': 'm01s03i236',
        'pr': 'm01s05i216',
        'gpp': 'm01s03i261',
        'npp': 'm01s03i262',
        'rh': 'm01s03i293',
        'landcflx': 'm01s03i326',
        'totcflx': 'm01s03i327',
        'cv': 'm01s19i002',
        'cs': 'm01s19i016',
        'dist': 'm01s19i012',
        'frac': 'm01s19i013',
        'ocn': 'm01s00i250',
        'emiss': 'm01s00i251',
        'co2': 'm01s00i252',
        'tos': 'm02s00i101',
        'sal': 'm02s00i102',
        'tco2': 'm02s00i103',
        'alk': 'm02s00i104',
        'nut': 'm02s00i105',
        'phy': 'm02s00i106',
        'zoo': 'm02s00i107',
        'detn': 'm02s00i108',
        'detc': 'm02s00i109',
        'pco2': 'm02s30i248',
        'fgco2': 'm02s30i249',
        'rlut': 'm01s02i205',
        'rlutcs': 'm01s02i206',
        'rsdt': 'm01s01i207',
        'rsut': 'm01s01i208',
        'rsutcs': 'm01s01i209',
    }
   
    return switcher.get(s, "nothing")
 
def stash_nc(s):
    switcher = {
        'tas': 3236,
        'pr': 5216,
        'gpp': 3261,
        'npp': 3262,
        'rh': 3293,
        'landcflx': 3326,
        'totcflx': 3327,
        'cv': 19002,
        'cs': 19016,
        'dist': 19012,
        'frac': 19013,
        'ocn': 250,
        'emiss': 251,
        'co2': 252,
        'tos': 101,
        'sal': 102,
        'tco2': 103,
        'alk': 104,
        'nut': 105,
        'phy': 106,
        'zoo': 107,
        'detn': 108,
        'detc': 109,
        'pco2': 30248,
        'fgco2': 30249,
        'rlut': 2205,
        'rlutcs': 2206,
        'rsdt': 1207,
        'rsut': 1208,
        'rsutcs': 1209,
    }
   
    return switcher.get(s, "nothing")

# Corrected two-letter month codes (UM-style)
MONTH_MAP_ALPHA = {
    "ja": 1,   # January
    "fb": 2,   # February
    "mr": 3,   # March
    "ar": 4,   # April
    "my": 5,   # May
    "jn": 6,   # June
    "jl": 7,   # July
    "ag": 8,   # August
    "sp": 9,   # September
    "ot": 10,  # October
    "nv": 11,  # November
    "dc": 12,  # December
}

def decode_month(mon_code: str) -> int:
    """
    Decode either:
      - 'dc' style codes
      - '11','21',...,'91','a1','b1','c1' style codes
    """
    if not mon_code:
        return 0
    s = mon_code.lower()

    # alpha codes: 'dc', 'sp', ...
    if s.isalpha():
        return MONTH_MAP_ALPHA.get(s, 0)

    # numeric/hex-ish codes: '11'..'91','a1','b1','c1'
    # rule: first char encodes month index; second char is typically '1' (ignore)
    # '1'..'9' => 1..9, 'a' => 10, 'b' => 11, 'c' => 12
    if len(s) == 2:
        first = s[0]
        if first.isdigit():
            m = int(first)
            return m if 1 <= m <= 9 else 0
        if first in ("a", "b", "c"):
            return {"a": 10, "b": 11, "c": 12}[first]

    return 0


def find_matching_files(expt_name, model, up, start_year=None, end_year=None, base_dir="~/dump2hold"):
    """
    Find matching data files for a given experiment and sort them by year/month.

    Supports:
      - xqhuja#pi000001853dc+
      - xqhujo#da00000185511+  (11=Jan, 91=Sep, a1=Oct, b1=Nov, c1=Dec)
    """
    base_dir = os.path.expanduser(base_dir)
    datam_path = os.path.join(base_dir, expt_name, "datam")
    if not os.path.isdir(datam_path):
        datam_path = base_dir
    # Month token can be either:
    #   - two letters: [a-zA-Z]{2}
    #   - two chars: [0-9a-cA-C][0-9]
    pattern = (
        fr"{re.escape(expt_name)}[{model}]\#{re.escape(up)}00000"
        fr"(\d{{4}})"                 # year
        fr"([a-zA-Z]{{2}}|[0-9a-cA-C][0-9])"  # month token
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
        mon_code = match.group(2)
        month = decode_month(mon_code)

        if month == 0:
            continue  # skip unparseable months

        if (start_year is None or year >= start_year) and (end_year is None or year <= end_year):
            matching_files.append((year, month, f))

    matching_files.sort(key=lambda x: (x[0], x[1]))
    return matching_files

# === Utility: compute masked terrestrial area ===
def compute_terrestrial_area(cube):
    if cube.ndim > 2:
        cube2d = cube[0]
    else:
        cube2d = cube
    for name in ("latitude", "longitude"):
        coord = cube2d.coord(name)
        if not coord.has_bounds():
            coord.guess_bounds()
    cellarea = cart.area_weights(cube2d)
    mask = np.ma.getmaskarray(cube2d.data)
    total_area = np.ma.array(cellarea, mask=mask).sum()
    return float(total_area)

# Initialise total_terrestrial_area

# Convert to PgC/year
var_dict = {
     'Ocean flux': (12/44)*3600*24*360*(1e-12),         # kgCO2/m2/s to PgC/yr
     'm01s00i250': (12/44)*3600*24*360*(1e-12),         # same as Ocean flux
     'field1560_mm_srf': (12/44)*3600*24*360*(1e-12),       # same as Ocean flux
     'GPP': 3600*24*360*(1e-12),                      # from kgC/m2/s to PgC/yr
     'NPP': 3600*24*360*(1e-12),                      # from kgC/m2/s to PgC/yr
     'P resp': 3600*24*360*(1e-12),                       # from kgC/m2/s to PgC/yr
     'S resp': 3600*24*360*(1e-12),                       # from kgC/m2/s to PgC/yr
     'litter flux': (1e-12),                           # from kgC/m2/yr to PgC/yr
     'V carb': (1e-12),                          # from kgC/m2 to PgC
     'vegetation_carbon_content': (1e-12),           # from kgC/m2 to PgC
     'S carb': (1e-12),                          # from kgC/m2 to PgC
     'soilCarbon': (1e-12),                          # from kgC/m2 to PgC
     'Air flux': (12)/1000*(1e-12),                           # from molC/m2/yr to PgC/yr
     'm02s30i249': (12)/1000*(1e-12),                           # same as Air flux
     'field646_mm_dpth': (12)/1000*(1e-12),                  # same as Air flux
     'Total co2': 28.97/44.01*(1e6),                    # from mmr to ppmv
     'm01s00i252': 28.97/44.01*(1e6),                    # same as Total co2
     'precip': 86400,                                   # from kg/m2/s to mm/day
     'Others': 1,                                       # no conversion
}

# === Utility: extract cube by STASH or stash_code ===
def _msi_from_stash_obj(st):
    """STASH(model=1, section=3, item=261) -> 'm01s03i261'"""
    if st is None:
        return None
    try:
        return f"m{int(st.model):02d}s{int(st.section):02d}i{int(st.item):03d}"
    except Exception:
        pass
    try:
        # some iris versions expose .msi
        s = str(st.msi).strip()
        if s.startswith("m") and "s" in s and "i" in s:
            return s
    except Exception:
        pass
    return None

def _msi_from_numeric_stash_code(code):
    """
    Numeric stash_code like 3261 -> s=3, i=261 -> 'm01s03i261'
    Heuristic for model:
      section >= 30 => model 2 (ocean/CO2 flux sections in your data)
      else => model 1
    """
    if code is None:
        return None
    try:
        n = int(code)  # handles np.int16 etc
    except Exception:
        return None
    section = n // 1000
    item = n % 1000
    model = 2 if section >= 30 else 1
    return f"m{model:02d}s{section:02d}i{item:03d}"

def _msi_from_any_attr(attrs):
    """Return MSI from either attrs['STASH'] or attrs['stash_code']."""
    if not attrs:
        return None
    # PP-style
    if "STASH" in attrs:
        msi = _msi_from_stash_obj(attrs.get("STASH"))
        if msi:
            return msi
    # NetCDF-style numeric
    if "stash_code" in attrs:
        msi = _msi_from_numeric_stash_code(attrs.get("stash_code"))
        if msi:
            return msi
    return None

def try_extract(cubes, code, stash_lookup_func=None, debug=False):
    """
    Extract cubes matching:
      - STASH object attribute, or
      - numeric stash_code attribute

    code can be:
      - MSI string 'm01s03i261'
      - short name like 'gpp' if stash_lookup_func provided (your stash()).
      - numeric stash_code like 3261
    """
    candidates = [code]

    # Expand short-name -> MSI using your mapping
    if stash_lookup_func is not None and isinstance(code, str):
        msi = stash_lookup_func(code)
        if msi and msi != "nothing":
            candidates.append(msi)

    # Add coercions
    try:
        candidates.append(str(code))
    except Exception:
        pass

    # If numeric-like, include int form
    if isinstance(code, (int, np.integer)) or (isinstance(code, str) and code.isdigit()):
        try:
            candidates.append(int(code))
        except Exception:
            pass

    # Normalise candidate MSIs
    cand_msi = set()
    for c in candidates:
        # MSI strings pass through
        if isinstance(c, str) and c.startswith("m") and "s" in c and "i" in c:
            cand_msi.add(c.strip())
            continue
        # numeric stash_code -> MSI
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

        return (cube_msi in cand_msi)

    try:
        return cubes.extract(Constraint(cube_func=_match))
    except Exception:
        return iris.cube.CubeList([])
    

# === Utility: compute global mean and scaled total ===
def global_total_pgC(cube, var):
    # ensure cube is not None
    if cube is None:
        raise ValueError("None cube passed to global_total_pgC()")
    # Handle CubeList input
    if isinstance(cube, iris.cube.CubeList):
        if not cube:
            raise ValueError("Empty CubeList passed to global_mean_pgC()")
        cube = cube[0]  # Get first cube in CubeList
    for name in ("latitude", "longitude"):
        coord = cube.coord(name)
        if not coord.has_bounds():
            coord.guess_bounds()
    weights = cart.area_weights(cube)
    gm = cube.collapsed(["latitude", "longitude"], iris.analysis.SUM, weights=weights)
    # Note here we hard-wired the factor to only include terrestrial area,
    # as we mainly deal with terrestrial carbon cycle variables.
    # if var == 'Total co2' or var == 'Others':
    #     total_terrestrial_area = 1
    # else:
    #     total_terrestrial_area = compute_terrestrial_area(cube)
    gm.data = gm.data * var_dict[var]
    return gm

def global_mean_pgC(cube, var):
    # ensure cube is not None
    if cube is None:
        raise ValueError("None cube passed to global_mean_pgC()")
    # Handle CubeList input
    if isinstance(cube, iris.cube.CubeList):
        if not cube:
            raise ValueError("Empty CubeList passed to global_mean_pgC()")
        cube = cube[0]  # Get first cube in CubeList
    for name in ("latitude", "longitude"):
        coord = cube.coord(name)
        if not coord.has_bounds():
            coord.guess_bounds()
    weights = cart.area_weights(cube)
    gm = cube.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=weights)
    # Note here we hard-wired the factor to only include terrestrial area,
    # as we mainly deal with terrestrial carbon cycle variables.
    # if var == 'Total co2' or var == 'Others':
    #     total_terrestrial_area = 1
    # else:
    #     total_terrestrial_area = compute_terrestrial_area(cube)
    gm.data = gm.data * var_dict[var]
    return gm

# for TRIFFID variables that have triffid time coord
def merge_monthly_results(results, require_full_year=False):
    """
    Merge multiple compute_monthly_mean()-style outputs (fractional years)
    into one annual-mean time series.

    Parameters
    ----------
    results : list[dict]
        Each dict has keys: 'years' (fractional years) and 'data'.
    require_full_year : bool
        If True, only return years with 12 months present.

    Returns
    -------
    dict
        {'years': integer years, 'data': annual mean values}
    """
    all_years, all_data = [], []
    for r in results:
        all_years.extend(r["years"])
        all_data.extend(r["data"])

    df = pd.DataFrame({"year_frac": all_years, "value": all_data})

    # Reconstruct integer year and month (same approach as your monthly merge)
    df["year"] = df["year_frac"].astype(int)
    df["month"] = np.round((df["year_frac"] - df["year"]) * 12).astype(int) + 1
    df.loc[df["month"] < 1, "month"] = 1
    df.loc[df["month"] > 12, "month"] = 12

    # 1) average duplicates at (year, month)
    monthly = df.groupby(["year", "month"], as_index=False)["value"].mean()

    # 2) annual mean across months
    annual = monthly.groupby("year")["value"].agg(["mean", "count"]).reset_index()

    if require_full_year:
        annual = annual[annual["count"] == 12]

    return {
        "years": annual["year"].to_numpy(dtype=int),
        "data": annual["mean"].to_numpy(),
    }


# === Utility: compute annual mean from scaled total ===
def compute_monthly_mean(cube, var):
    """
    Compute area-weighted annual means from an Iris cube.
    Handles 360_day calendars and missing bounds safely.
    Returns (years, annual_means).
    """
    gm = global_total_pgC(cube, var)

    # --- Get time coordinate robustly ---
    time_coords = [c for c in gm.coords() if c.standard_name == 'time' or c.name() in ('t', 'time', 'TIME')]
    if not time_coords:
        raise ValueError("❌ No valid time coordinate found in cube.")
    tcoord = time_coords[0]

    # --- Ensure calendar and units are valid ---
    if not getattr(tcoord.units, "calendar", None):
        tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")
    elif str(tcoord.units).startswith("unknown"):
        tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")

    # --- Convert time → datetimes ---
    times = tcoord.units.num2date(tcoord.points)

    # --- Build DataFrame to group by year/month ---
    df = pd.DataFrame({
        "year": [t.year for t in times],
        "month": [t.month for t in times],
        "value": gm.data
    })

    # --- Average duplicates within each (year, month) ---
    df_monthly = df.groupby(["year", "month"], as_index=False)["value"].mean()

    # --- Convert to fractional year ---
    df_monthly["year_frac"] = df_monthly["year"] + (df_monthly["month"] - 1) / 12

    return {
        "years": df_monthly['year_frac'].to_numpy(),
        "data": df_monthly['value'].to_numpy(),
        "name": cube.long_name or cube.standard_name or var,
        "units": str(gm.units)
    }

# === Utility: compute annual mean from scaled total ===
def compute_annual_mean(cube, var):
    """
    Compute area-weighted annual means from an Iris cube.
    Handles 360_day calendars and missing bounds safely.
    Returns (years, annual_means).
    """
    if var == 'Others':
        gm = global_mean_pgC(cube, var)
    else:
        gm = global_total_pgC(cube, var)

    # --- Get time coordinate robustly ---
    time_coords = [c for c in gm.coords() if c.standard_name == 'time' or c.name() in ('t', 'time', 'TIME')]
    if not time_coords:
        raise ValueError("❌ No valid time coordinate found in cube.")
    tcoord = time_coords[0]

    # --- Ensure calendar and units are valid ---
    if not getattr(tcoord.units, "calendar", None):
        tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")
    elif str(tcoord.units).startswith("unknown"):
        tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")

    # --- Convert time → datetimes & extract years ---
    times = tcoord.units.num2date(tcoord.points)
    years = np.array([t.year for t in times])

    # --- Compute annual mean ---
    unique_years, idx = np.unique(years, return_inverse=True)
    # annual_means = np.array([np.nanmean(gm.data[idx == i]) for i in range(len(unique_years))])
    if gm.data.ndim == 0:  # scalar (no time dimension)
        annual_means = np.repeat(gm.data.item(), len(unique_years))
    else:
        annual_means = np.array([
            np.nanmean(gm.data[idx == i]) for i in range(len(unique_years))
    ])
    return {
        "years": unique_years,
        "data": annual_means,
        "name": cube.long_name or cube.standard_name or var,
        "units": str(gm.units)
    }

def load_reccap_mask():
    reccap_mask = iris.load_cube(
        os.path.expanduser(
            '~/scripts/hadcm3b-ensemble-validator/observations/RECCAP_AfricaSplit_MASK11_Mask_regridded.hadcm3bl_grid.nc'))

    regions = {
        1: "North_America",
        2: "South_America",
        3: "Europe",
        4: "Africa",  # combine North Africa (=4) and South Africa (+5)
        6: "North_Asia",
        7: "Central_Asia",
        8: "East_Asia",
        9: "South_Asia",
        10: "South_East_Asia",
        11: "Oceania",
    }
    return reccap_mask, regions

def region_mask(region):
    reccap_mask, regions = load_reccap_mask()

    # Find correct ID(s) for the region
    if region == "Africa":
        target_ids = [4, 5]
    else:
        target_ids = [k for k, v in regions.items() if v == region]
        if not target_ids:
            raise ValueError(f"Region '{region}' not found in RECCAP regions.")
    
    # Build binary mask
    mask = reccap_mask.copy()
    mask.data = np.isin(mask.data, target_ids).astype(int)
    return mask

# === Utility: compute annual mean from scaled total ===
# def compute_regional_annual_mean(cube, var, region):
#     """
#     Compute area-weighted annual means from an Iris cube.
#     Handles 360_day calendars and missing bounds safely.
#     Returns (years, annual_means).
#     """
#     # Handle CubeList input
#     if isinstance(cube, iris.cube.CubeList):
#         if not cube:
#             raise ValueError("Empty CubeList passed to global_mean_pgC()")
#         cube = cube[0]  # Get first cube in CubeList
#     for name in ("latitude", "longitude"):
#         coord = cube.coord(name)
#         if not coord.has_bounds():
#             coord.guess_bounds()
#     cube = cube.copy()
#     # --- Apply region mask ---
#     if region != 'global':
#         cube = cube * region_mask(region)
#     else:
#         pass
#     # --- Compute regional mean or total based on variable ---
#     if var == 'Others':
#         gm = global_mean_pgC(cube, var)
#     else:
#         gm = global_total_pgC(cube, var)

#     # --- Get time coordinate robustly ---
#     time_coords = [c for c in gm.coords() if c.standard_name == 'time' or c.name() in ('t', 'time', 'TIME')]
#     if not time_coords:
#         raise ValueError("❌ No valid time coordinate found in cube.")
#     tcoord = time_coords[0]

#     # --- Ensure calendar and units are valid ---
#     if not getattr(tcoord.units, "calendar", None):
#         tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")
#     elif str(tcoord.units).startswith("unknown"):
#         tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")

#     # --- Convert time → datetimes & extract years ---
#     times = tcoord.units.num2date(tcoord.points)
#     years = np.array([t.year for t in times])

#     # --- Compute annual mean ---
#     unique_years, idx = np.unique(years, return_inverse=True)
#     # annual_means = np.array([np.nanmean(gm.data[idx == i]) for i in range(len(unique_years))])
#     if gm.data.ndim == 0:  # scalar (no time dimension)
#         annual_means = np.repeat(gm.data.item(), len(unique_years))
#     else:
#         annual_means = np.array([
#             np.nanmean(gm.data[idx == i]) for i in range(len(unique_years))
#     ])
#     return {
#         "years": unique_years,
#         "data": annual_means,
#         "name": cube.long_name or cube.standard_name or var,
#         "units": str(gm.units),
#         "region": region,
#     }

def compute_regional_annual_mean(cube, var, region):
    """
    Compute area-weighted annual means from an Iris cube.
    Handles 360_day calendars and missing bounds safely.
    Returns dict with years and annual mean data.
    """
    # --- Handle CubeList input ---
    if isinstance(cube, iris.cube.CubeList):
        if not cube:
            raise ValueError("Empty CubeList passed to compute_regional_annual_mean_test()")
        cube = cube[0]

    cube = cube.copy()

    # --- Ensure bounds for area weights ---
    for name in ("latitude", "longitude"):
        coord = cube.coord(name)
        if not coord.has_bounds():
            coord.guess_bounds()

    # --- Build weights, and mask weights for region ---
    weights = cart.area_weights(cube)

    if region != "global":
        m_obj = region_mask(region)
        m2d = np.asarray(m_obj.data) if isinstance(m_obj, iris.cube.Cube) else np.asarray(m_obj)
        m2d = np.squeeze(m2d)
        # Sanity check: mask must match the horizontal grid
        if m2d.shape != weights.shape[-2:]:
            raise ValueError(f"Mask shape {m2d.shape} != weights lat/lon shape {weights.shape[-2:]}")
        # Handle pfts broadcasting
        if cube.ndim == 4:
            m = m2d[None, None, :, :]
        else:
            m = m2d[None, :, :]
        w = weights * m
    else:
        w = weights

    # Important: avoid printing huge arrays; just scalar summaries
    # print(f"weights.sum() (global grid): {weights.sum():.6e}")
    # if region != "global":
    #     print(f"masked weights.sum() ({region}): {w.sum():.6e}")

    # --- Compute regional mean or total based on variable ---
    # IMPORTANT: do the collapse HERE, using w, not inside other functions.
    if var in ("Others", "precip"):
        gm = cube.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=w)
    else:
        gm = cube.collapsed(["latitude", "longitude"], iris.analysis.SUM, weights=w)

    # Apply scaling after collapse (cheap)
    gm.data = gm.data * var_dict[var]

    # --- Get time coordinate robustly ---
    time_coords = [c for c in gm.coords() if c.standard_name == "time" or c.name() in ("t", "time", "TIME")]
    if not time_coords:
        raise ValueError("No valid time coordinate found in collapsed cube.")
    tcoord = time_coords[0]

    # --- Ensure calendar and units are valid ---
    if not getattr(tcoord.units, "calendar", None) or str(tcoord.units).startswith("unknown"):
        tcoord.units = cf_units.Unit("days since 1850-12-01 00:00:00", calendar="360_day")

    # --- Convert time → years ---
    times = tcoord.units.num2date(tcoord.points)
    years = np.array([t.year for t in times])

    unique_years, idx = np.unique(years, return_inverse=True)

    # Force realisation only on the 1D time series (prevents “background” lazy compute later)
    gm_series = np.asarray(gm.data)

    if gm_series.ndim == 0:
        annual_means = np.repeat(gm_series.item(), len(unique_years))
    else:
        annual_means = np.array([np.nanmean(gm_series[idx == i]) for i in range(len(unique_years))])

    return {
        "years": unique_years,
        "data": annual_means,
        "name": cube.long_name or cube.standard_name or var,
        "units": str(gm.units),
        "region": region,
    }


# new func 02
def extract_annual_means(expts_list, var_list=None, var_mapping=None, regions=None):

    default_var_list = ['soilResp', 'soilCarbon', 'VegCarb', 'fracPFTs', 'GPP', 'NPP', 'fgco2', 'temp', 'precip']
    default_var_mapping = var_mapping = ['S resp','S carb','V carb','Others',
               'GPP','NPP','field646_mm_dpth', 'Others', 'precip']
    if var_list is None:
        var_list = default_var_list

    # Generate default or custom var_mapping
    if var_mapping is None:
        var_mapping = default_var_mapping
    # else:
    #     var_mapping = [v if v in var_dict else 'Others' for v in var_mapping]

    dict_annual_means = {}
    dict_frac = {}
    dict_temp = {}
    dict_precip = {}

    _, regions_dict = load_reccap_mask()
    all_regions = list(regions_dict.values()) + ['Africa', 'global']
    target_regions = regions if regions is not None else all_regions

    for expt in expts_list:
        base_dir = f'~/annual_mean/{expt}/'
        dict_annual_means[expt] = {}
        base_dir = os.path.expanduser(base_dir)
        os.makedirs(base_dir, exist_ok=True)

        filenames = glob.glob(os.path.join(base_dir, "**/*.nc"), recursive=True)
        cubes = iris.load(filenames)

        sr = try_extract(cubes, 'rh')
        sc = try_extract(cubes, 'cs')
        vc = try_extract(cubes, 'cv')

        frac = try_extract(cubes, 'frac')
        if not frac:
            print("❌ 'frac' variable not found, trying stash code 3317")
            frac = try_extract(cubes, 3317)
            if not frac:
                print("❌ 'frac' variable still not found.")

        gpp = try_extract(cubes, 'gpp')
        npp = try_extract(cubes, 'npp')
        fgco2 = try_extract(cubes, 'fgco2')
        temp = try_extract(cubes, 'tas')
        precip = try_extract(cubes, 'pr')

        dict_frac[expt] = frac
        dict_temp[expt] = temp
        dict_precip[expt] = precip

        cube_list = [sr, sc, vc, frac, gpp, npp, fgco2, temp, precip]

        for region in target_regions:
            dict_annual_means[expt][region] = {}
            for i, (cubeset, varname, mapping) in enumerate(zip(cube_list, var_list, var_mapping)):
                if not cubeset:
                    continue
                if varname == 'fgco2' and region != 'global':
                    continue  # Skip fgco2 for non-global regions

                cube = cubeset[0]
                # Handle regular variables
                if cube is not None and varname != 'fracPFTs':
                    output = compute_regional_annual_mean(cube, mapping, region)
                    output['units'] = {
                        'temp': 'K',
                        'precip': 'mm/day'
                    }.get(varname, 'PgC/year')
                    dict_annual_means[expt][region][varname] = output
                # Handle fracPFTs separately
                else:
                    frac_data = {}
                    for j in range(1, 10):
                        try:
                            frac_pft = cube.extract(Constraint(coord_values={'generic': j}))
                            output = compute_regional_annual_mean(frac_pft, mapping, region)
                            frac_data[f'PFT {j}'] = output
                        except:
                            continue
                    dict_annual_means[expt][region][varname] = frac_data

            # NEP
            if 'NPP' in dict_annual_means[expt][region] and 'soilResp' in dict_annual_means[expt][region]:
                nep_years = dict_annual_means[expt][region]['NPP']['years'].copy()
                nep_data = dict_annual_means[expt][region]['NPP']['data'] - dict_annual_means[expt][region]['soilResp']['data']
                dict_annual_means[expt][region]['NEP'] = {
                    'years': nep_years,
                    'data': nep_data,
                    'name': 'Net Ecosystem Production',
                    'units': dict_annual_means[expt][region]['NPP']['units'],
                    'region': region
                }

            # Land Carbon
            if all(k in dict_annual_means[expt][region] for k in ['soilCarbon', 'VegCarb', 'NEP']):
                lc_years = dict_annual_means[expt][region]['NEP']['years'].copy()
                lc_data = dict_annual_means[expt][region]['soilCarbon']['data'] + \
                          dict_annual_means[expt][region]['VegCarb']['data'] + \
                          dict_annual_means[expt][region]['NEP']['data']
                dict_annual_means[expt][region]['Land Carbon'] = {
                    'years': lc_years,
                    'data': lc_data,
                    'name': 'Total Land Carbon',
                    'units': dict_annual_means[expt][region]['NPP']['units'],
                    'region': region
                }

            # Trees Total
            if 'fracPFTs' in dict_annual_means[expt][region] and \
               all(p in dict_annual_means[expt][region]['fracPFTs'] for p in ['PFT 1', 'PFT 2']):
                tree_years = dict_annual_means[expt][region]['fracPFTs']['PFT 1']['years'].copy()
                tree_data = dict_annual_means[expt][region]['fracPFTs']['PFT 1']['data'] + \
                            dict_annual_means[expt][region]['fracPFTs']['PFT 2']['data']
                dict_annual_means[expt][region]['Trees Total'] = {
                    'years': tree_years,
                    'data': tree_data,
                    'name': 'Total Trees Fraction',
                    'units': 'fraction',
                    'region': region
                }
    # print(var_list, var_mapping, var_dict)
    return dict_annual_means

def extract_annual_means_to_zarr(recipe_path: Path) -> None:
    """Temporary shim: loads recipe and calls your existing `extract_annual_means`.

    Next steps:
      - replace dict outputs with xarray Dataset
      - write to Zarr/NetCDF in cache_root
    """
    with open(recipe_path, "r") as f:
        recipe = yaml.safe_load(f)

    expts = recipe.get("expts", [])
    # call into your existing prototype function (still returns dict)
    from .annual_means import extract_annual_means as _extract

    out = _extract(expts_list=expts, regions=recipe.get("preprocess", {}).get("regions"))
    # TODO: convert `out` to xarray Dataset and persist to Zarr
    print(f"Extracted annual means for {len(expts)} experiments (prototype dict output).")
