from __future__ import annotations
import typer
from pathlib import Path
from ccval.preprocess.annual_means import extract_annual_means_to_zarr

app = typer.Typer(add_completion=False)

@app.command()
def preprocess(recipe: Path):
    """Run preprocessing from a YAML recipe."""
    extract_annual_means_to_zarr(recipe)

if __name__ == "__main__":
    app()
