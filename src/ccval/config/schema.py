from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Mapping, Optional

@dataclass(frozen=True)
class PathsConfig:
    raw_root: Path
    cache_root: Path
    output_root: Path

@dataclass(frozen=True)
class PreprocessConfig:
    regions: Sequence[str] = ("global",)
    n_years: Optional[int] = None  # if you want to trim early years
    regrid_target: Optional[str] = None  # TODO: define grid spec

@dataclass(frozen=True)
class Recipe:
    name: str
    expts: Sequence[str]
    paths: PathsConfig
    preprocess: PreprocessConfig
    variables: Mapping[str, str]  # user var -> internal mapping (e.g. 'soilResp' -> 'S resp')
