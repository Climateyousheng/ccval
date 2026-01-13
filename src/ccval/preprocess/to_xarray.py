from __future__ import annotations

from typing import Any, Dict, Tuple, List
import numpy as np
import xarray as xr


def annual_means_dict_to_xr(
    annual: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    year_name: str = "year",
    experiment_name: str = "experiment",
    region_name: str = "region",
    pft_name: str = "pft",
    frac_key: str = "fracPFTs",
) -> xr.Dataset:
    """
    Convert the output of `extract_annual_means()` into an xarray.Dataset.

    Parameters
    ----------
    annual
        Nested dict: annual[expt][region][var] -> dict with keys {"years","data",...}
        Special case: annual[expt][region]["fracPFTs"][pft_label] -> dict with {"years","data",...}
    """

    # --- coordinates
    experiments = sorted(annual.keys())
    regions = sorted({r for expt in annual for r in annual[expt].keys()})

    # Collect all years across all variables/regions/experiments
    year_set = set()
    pft_set = set()

    for expt in experiments:
        for region in annual.get(expt, {}):
            for var, payload in annual[expt][region].items():
                if var == frac_key and isinstance(payload, dict):
                    # payload: {"PFT 1": {...}, "PFT 2": {...}, ...}
                    for pft_label, pft_payload in payload.items():
                        pft_set.add(str(pft_label))
                        yrs = pft_payload.get("years", [])
                        year_set.update([int(y) for y in np.asarray(yrs).astype(int)])
                else:
                    if isinstance(payload, dict) and "years" in payload:
                        yrs = payload.get("years", [])
                        year_set.update([int(y) for y in np.asarray(yrs).astype(int)])

    years = np.array(sorted(year_set), dtype=int)
    pfts = sorted(pft_set)

    # Collect variable names (excluding frac_key because we store it separately)
    var_names = sorted({
        var
        for expt in experiments
        for region in annual.get(expt, {})
        for var in annual[expt][region].keys()
        if var != frac_key
    })

    # Helper: index lookup
    exp_index = {e: i for i, e in enumerate(experiments)}
    reg_index = {r: j for j, r in enumerate(regions)}
    year_index = {int(y): k for k, y in enumerate(years)}
    pft_index = {p: i for i, p in enumerate(pfts)}

    data_vars = {}
    var_units = {}

    # --- allocate and fill non-PFT variables: (experiment, region, year)
    for var in var_names:
        arr = np.full((len(experiments), len(regions), len(years)), np.nan, dtype=float)
        units = None

        for expt in experiments:
            for region in regions:
                payload = annual.get(expt, {}).get(region, {}).get(var)
                if not payload or not isinstance(payload, dict):
                    continue

                yrs = payload.get("years", None)
                vals = payload.get("data", None)
                if yrs is None or vals is None:
                    continue

                yrs = np.asarray(yrs).astype(int)
                vals = np.asarray(vals, dtype=float)

                ei = exp_index[expt]
                rj = reg_index[region]
                for y, v in zip(yrs, vals):
                    yi = year_index.get(int(y))
                    if yi is not None:
                        arr[ei, rj, yi] = v

                if units is None:
                    units = payload.get("units", None)

        da = xr.DataArray(
            arr,
            dims=(experiment_name, region_name, year_name),
            coords={
                experiment_name: experiments,
                region_name: regions,
                year_name: years,
            },
            name=_safe_var_name(var),
        )
        if units:
            da.attrs["units"] = units
        data_vars[da.name] = da
        var_units[da.name] = units

    # --- fracPFTs: (experiment, region, year, pft)
    if pfts:
        frac_arr = np.full(
            (len(experiments), len(regions), len(years), len(pfts)),
            np.nan,
            dtype=float
        )
        frac_units = None

        for expt in experiments:
            for region in regions:
                payload = annual.get(expt, {}).get(region, {}).get(frac_key)
                if not payload or not isinstance(payload, dict):
                    continue
                for pft_label, pft_payload in payload.items():
                    pft_label = str(pft_label)
                    if pft_label not in pft_index:
                        continue

                    yrs = pft_payload.get("years", None)
                    vals = pft_payload.get("data", None)
                    if yrs is None or vals is None:
                        continue

                    yrs = np.asarray(yrs).astype(int)
                    vals = np.asarray(vals, dtype=float)

                    ei = exp_index[expt]
                    rj = reg_index[region]
                    pi = pft_index[pft_label]
                    for y, v in zip(yrs, vals):
                        yi = year_index.get(int(y))
                        if yi is not None:
                            frac_arr[ei, rj, yi, pi] = v

                    if frac_units is None:
                        frac_units = pft_payload.get("units", None)

        frac_da = xr.DataArray(
            frac_arr,
            dims=(experiment_name, region_name, year_name, pft_name),
            coords={
                experiment_name: experiments,
                region_name: regions,
                year_name: years,
                pft_name: pfts,
            },
            name=frac_key,
        )
        if frac_units:
            frac_da.attrs["units"] = frac_units
        data_vars[frac_da.name] = frac_da

    ds = xr.Dataset(data_vars)

    # Optional: attach a little provenance / notes
    ds.attrs["ccval_note"] = "Converted from CCVal annual-means nested dict."
    ds.attrs["dims"] = ", ".join(ds.dims.keys())

    return ds


def _safe_var_name(name: str) -> str:
    """
    Make variable name xarray/NetCDF-friendly but keep readability.
    """
    s = name.strip()
    # keep alnum, underscore; convert spaces and hyphens to underscore
    s = s.replace(" ", "_").replace("-", "_")
    # avoid weird characters (e.g. "/")
    for ch in ["/", "(", ")", "[", "]", "{", "}", ":", ";", ","]:
        s = s.replace(ch, "")
    # collapse double underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s
