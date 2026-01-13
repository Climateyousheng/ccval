# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development

```bash
# Install in editable mode
pip install -e .

# Run CLI
ccval --help

# Run preprocessing with a recipe
ccval preprocess workflows/recipes/preprocess_land.yml

# Run tests
pytest tests/
```

## Architecture

CCVal is a land carbon-cycle diagnostics tool for HPC model outputs, built around Iris cubes and YAML-driven workflows.

### Module Structure

- **io/**: Data loading via Iris (`iris_loaders.py`) - handles UM PP files and NetCDF with STASH code extraction
- **preprocess/**: Core analysis (`annual_means.py`) - computes area-weighted annual/regional means, unit conversions (kgC→PgC), and derived variables (NEP, Land Carbon)
- **viz/**: Plotting (`timeseries.py` for time series/pie charts, `maps.py` for spatial difference maps)
- **cli/**: Typer-based CLI that dispatches to preprocessing functions based on YAML recipes
- **config/**: Dataclass schemas for recipes (`PathsConfig`, `PreprocessConfig`, `Recipe`)

### Key Patterns

**STASH Code Extraction**: Variables are identified by UM STASH codes (e.g., `m01s03i261` for GPP). The `try_extract()` function in `annual_means.py` handles both PP-style `STASH` attributes and NetCDF-style numeric `stash_code` attributes.

**Unit Conversion**: `var_dict` maps variable names to conversion factors (e.g., `kgC/m2/s → PgC/yr`). New variables need entries here.

**Regional Analysis**: Uses RECCAP mask for regional breakdowns. The `compute_regional_annual_mean()` function applies masked area weights.

**Recipe-Driven Workflows**: YAML files in `workflows/recipes/` define experiments, paths, regions, and variable mappings. The CLI loads these and passes to preprocessing functions.

### Data Flow

1. Recipe YAML specifies experiments and variables
2. `extract_annual_means()` loads pre-computed annual-mean NetCDF files from `~/annual_mean/{expt}/`
3. Cubes are extracted by STASH code, area-weighted, and aggregated by region/year
4. Output is a nested dict: `{expt: {region: {var: {years, data, units}}}}`
5. Visualization functions consume this dict structure
