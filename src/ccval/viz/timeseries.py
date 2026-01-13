# NOTE: Imported from user prototype 'plot.py'. Refactor progressively.
# analysis_utils.py

import os
import numpy as np
import matplotlib.pyplot as plt
import iris

legend_labels = {
    "xqhsh": "PI LU COU spinup",
    "xqhuc": "PI HadCM3 spinup",
}

color_map = {
    "xqhsh": "k",
    "xqhuc": "r",
}

# Group variables by their common prefix
def group_vars_by_prefix(data, expts_list=None, region="global", exclude=("fracPFTs",)):
    """
    data: dict[expt][region][var] -> series dict
    Returns: dict[prefix] -> sorted list of vars
    """
    grouped = {}

    if expts_list is None:
        expts_list = list(data.keys())

    for expt in expts_list:
        expt_block = data.get(expt, {})
        region_block = expt_block.get(region, {})
        for var in region_block.keys():
            if any(var.startswith(p) for p in exclude):
                continue
            prefix = var.split("_")[0] if "_" in var else var
            grouped.setdefault(prefix, set()).add(var)

    return {k: sorted(v) for k, v in grouped.items()}

# Plot time series of all variables grouped by prefix, excluding pfts
# Plot all variables in one figure with subplots
def plot_timeseries_grouped(data, expts_list, region, outdir,
                                legend_labels=None, color_map=None, show: bool = False,
                                exclude=("fracPFTs",), ncols=3):
    """
    data: dict[expt][region][var] -> {"years":..., "data":..., "units":...}
    """
    legend_labels = legend_labels or {}
    color_map = color_map or {}

    grouped = group_vars_by_prefix(data, expts_list=expts_list, region=region, exclude=exclude)

    os.makedirs(outdir, exist_ok=True)

    # Flatten all grouped variables for this region into one figure
    all_varnames = [var for group in grouped.values() for var in group]
    n_vars = len(all_varnames)
    if n_vars == 0:
        return

    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        sharex=True,
        dpi=100
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, var in zip(axes, all_varnames):
        any_series = None  # capture one valid series for units/title

        for exp in expts_list:
            series = data.get(exp, {}).get(region, {}).get(var)
            if not series:
                continue

            any_series = series
            label = legend_labels.get(exp, exp)

            years = np.asarray(series["years"])
            vals  = np.asarray(series["data"])

            ax.plot(
                years[1:], vals[1:],
                label=label,
                color=color_map.get(exp, "0.5"),
                lw=0.8
            )

        units = (any_series or {}).get("units", "")
        ax.set_title(f"{var} ({units}) {region}" if units else var, fontsize=9)
        ax.set_ylabel(units or "Value", fontsize=8)
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=6, loc="upper left")

    for k in range(len(all_varnames), len(axes)):
        fig.delaxes(axes[k])

    if len(all_varnames) > 0:
        axes[0].set_xlabel("Year", fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(outdir, f"allvars_{region}_timeseries.png")
    plt.savefig(outpath, dpi=300)
    if show:
        plt.show()
    plt.close()

# plot a single pie chart for one variable, one experiment, one year
def plot_regional_pie(data, varname, expt, year, outdir, legend_labels=None, show: bool = False):
    """
    Pie chart of a variable across regions for one experiment & year.

    data: dict[expt][region][var] -> series dict
    varname: e.g. "soilResp" (NOT "soilResp_North_America" anymore)
    """
    legend_labels = legend_labels or {}

    expt_block = data.get(expt, {})
    if not expt_block:
        raise ValueError(f"No data found for experiment '{expt}'")

    labels, values = [], []
    units = None

    for region, region_block in expt_block.items():
        series = region_block.get(varname)
        if not series:
            continue

        years = np.asarray(series["years"])
        vals  = np.asarray(series["data"])
        if year not in years:
            continue

        idx = np.where(years == year)[0][0]
        val = vals[idx]
        if val is None or np.isnan(val):
            continue

        units = units or series.get("units", "")
        labels.append(region.replace("_", " "))
        values.append(val)

    if not values:
        raise ValueError(f"No valid values found for {varname} in year {year} for {expt}")

    total = float(np.sum(values))
    labels_with_values = [
        f"{lab}\n{v:.2f} ({(v/total*100):.1f}%)"
        for lab, v in zip(labels, values)
    ]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        values,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.45),
        labels=labels_with_values,
        textprops=dict(color="black", fontsize=8),
    )

    title_label = legend_labels.get(expt, expt)
    unit_str = f" ({units})" if units else ""
    ax.set_title(f"{varname}{unit_str} distribution in {year}\n{title_label}", fontsize=10)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{varname}_regional_pie_{expt}_{year}.png")
    plt.savefig(outpath, dpi=300)
    if show:
        plt.show()
    plt.close()

# plot PFT fraction timeseries for one region
def plot_pft_timeseries(data, expts_list, region, outdir,
                               legend_labels=None, color_map=None, show: bool = False,
                               pfts=(1, 2, 3, 4, 5)):
    """
    Plot PFT fraction timeseries (PFT 1–5 by default) for one region.
    One figure per region containing 5 subplots (PFT1..PFT5),
    with multiple experiments overlaid in each subplot.

    data: dict[expt][region]["fracPFTs"]["PFT n"] -> {"years","data","units",...}
    """
    legend_labels = legend_labels or {}
    color_map = color_map or {}
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, dpi=100)
    axes = axes.flatten()

    # We'll use only first 5 axes, delete the 6th
    for ax, p in zip(axes, pfts):
        pft_key = f"PFT {p}"
        any_series = None

        for exp in expts_list:
            series = (
                data.get(exp, {})
                    .get(region, {})
                    .get("fracPFTs", {})
                    .get(pft_key)
            )
            if not series:
                continue

            any_series = series
            years = np.asarray(series["years"])
            vals  = np.asarray(series["data"])
            label = legend_labels.get(exp, exp)

            ax.plot(
                years[1:], vals[1:],  # keep your convention of dropping first entry
                label=label,
                color=color_map.get(exp, "0.5"),
                lw=0.9
            )

        units = (any_series or {}).get("units", "")
        ax.set_title(f"{pft_key} ({units})" if units else pft_key, fontsize=9)
        ax.set_ylabel(units or "Fraction", fontsize=8)
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=6, loc="upper left")

    # Remove extra subplot (6th)
    if len(axes) > len(pfts):
        fig.delaxes(axes[len(pfts)])

    axes[0].set_xlabel("Year", fontsize=9)
    plt.suptitle(f"PFT fractions (first 5) – {region}", fontsize=11, y=1.02)
    plt.tight_layout()

    outpath = os.path.join(outdir, f"fracPFTs_{region}_timeseries.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

# plot side-by-side pie charts for multiple experiments for one variable, one year
def plot_regional_pies(data, varname, expts_list, year, outdir,
                       legend_labels=None, show: bool = False):
    """
    Side-by-side pie charts of a variable across regions
    for multiple experiments in a given year.

    data: dict[expt][region][var] -> series dict
    varname: e.g. "soilResp"
    expts_list: list of experiments
    """
    legend_labels = legend_labels or {}

    n = len(expts_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), squeeze=False, dpi=100)
    axes = axes[0]

    for ax, expt in zip(axes, expts_list):
        expt_block = data.get(expt, {})
        if not expt_block:
            ax.set_title(f"{expt}\n(no data)")
            ax.axis("off")
            continue

        labels, values = [], []
        units = None

        for region, region_block in expt_block.items():
            series = region_block.get(varname)
            if not series:
                continue

            years = np.asarray(series["years"])
            vals  = np.asarray(series["data"])

            if year not in years:
                continue

            idx = np.where(years == year)[0][0]
            val = vals[idx]
            if val is None or np.isnan(val):
                continue

            units = units or series.get("units", "")
            labels.append(region.replace("_", " "))
            values.append(val)

        if not values:
            ax.set_title(f"{legend_labels.get(expt, expt)}\n(no valid data)")
            ax.axis("off")
            continue

        total = float(np.sum(values))
        labels_with_values = [
            f"{lab}\n{v:.2f} ({(v / total * 100):.1f}%)"
            for lab, v in zip(labels, values)
        ]

        ax.pie(
            values,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.45),
            labels=labels_with_values,
            textprops=dict(color="black", fontsize=8),
        )

        title_label = legend_labels.get(expt, expt)
        unit_str = f" ({units})" if units else ""
        ax.set_title(f"{title_label}\n{varname}{unit_str}\n{year}", fontsize=10)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{varname}_regional_pies_{year}.png")
    plt.savefig(outpath, dpi=300)
    if show:
        plt.show()
    plt.close()

# plot grouped bar charts for PFT fractions by region
def plot_pft_grouped_bars(
    data,
    expts_list,
    year,
    outdir,
    legend_labels=None,
    color_map=None,
    pfts=(1, 2, 3, 4, 5),
    show: bool = False,
):
    """
    Plot PFT 1–5 in a single figure.
    Each subplot = one PFT.
    X-axis = regions.
    Bars = experiments (different colours).

    data: dict[expt][region]["fracPFTs"]["PFT n"] -> series dict
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    legend_labels = legend_labels or {}
    color_map = color_map or {}
    os.makedirs(outdir, exist_ok=True)

    # --- determine consistent region ordering ---
    all_regions = sorted({
        region
        for expt in expts_list
        for region in data.get(expt, {}).keys()
    })

    n_regions = len(all_regions)
    n_expts = len(expts_list)
    bar_width = 0.8 / n_expts
    x = np.arange(n_regions)

    # --- layout: 2 rows × 3 columns (last panel empty) ---
    fig, axes = plt.subplots(2, 3, figsize=(1.2 * n_regions + 6, 7), dpi=100)
    axes = axes.flatten()

    for ax, p in zip(axes, pfts):
        pft_key = f"PFT {p}"

        for i, expt in enumerate(expts_list):
            values = []

            for region in all_regions:
                series = (
                    data.get(expt, {})
                        .get(region, {})
                        .get("fracPFTs", {})
                        .get(pft_key)
                )

                if not series:
                    values.append(np.nan)
                    continue

                years = np.asarray(series["years"])
                vals  = np.asarray(series["data"])
                if year not in years:
                    values.append(np.nan)
                    continue

                idx = np.where(years == year)[0][0]
                values.append(vals[idx])

            ax.bar(
                x + i * bar_width,
                values,
                width=bar_width,
                label=legend_labels.get(expt, expt),
                color=color_map.get(expt, None),
            )

        ax.set_title(pft_key, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.5)

        ax.set_xticks(x + bar_width * (n_expts - 1) / 2)
        ax.set_xticklabels(
            [r.replace("_", " ") for r in all_regions],
            rotation=45,
            ha="right",
            fontsize=8,
        )

    # --- remove unused subplot (6th) ---
    for k in range(len(pfts), len(axes)):
        fig.delaxes(axes[k])

    # --- shared labels & legend ---
    axes[0].set_ylabel("Fraction", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(expts_list),
               frameon=False, fontsize=9)

    fig.suptitle(f"PFT fractions by region ({year})", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    outpath = os.path.join(outdir, f"fracPFTs_1to5_grouped_bars_{year}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
