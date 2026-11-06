"""Load Argo float metadata from the local GDAC mirror and identify SIO floats."""
import glob
import os
from functools import cache
from pathlib import Path

import pandas as pd
import xarray as xr

GDAC = "/shared/argo/mirror"   # folder on wombat containing ar_index_global_prof.txt

# argo-maps/src/argo_maps/metadata.py -> parents[2] is the argo-maps project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

FIELDS = ["PROJECT_NAME", "PI_NAME", "OPERATING_INSTITUTION", "FLOAT_OWNER",
          "PLATFORM_TYPE", "PLATFORM_MAKER", "FLOAT_SERIAL_NO",
          "LAUNCH_DATE", "END_MISSION_DATE", "END_MISSION_STATUS", "ANOMALY"]

# Define patterns to search by institution and project name
SIO_PATTERN = r"\bSIO\b|Scripps"
INST_COLS = ["OPERATING_INSTITUTION", "PI_NAME", "FLOAT_OWNER"]

FFST_PATTERN = r"\bFFST\b"
PROJ_COLS = ["PROJECT_NAME"]

def _as_text(x) -> str:
    """Decode bytes to str; leave anything else as str(x)."""
    if isinstance(x, bytes):
        return x.decode("utf-8", "ignore")
    return str(x)

def get_str(ds, name):
    """Return a netCDF text variable as a clean Python string ('' if absent)."""
    if name not in ds.variables:
        return ""
    v = ds[name].values.ravel()
    if v.size == 0:
        return ""
    if v.dtype.kind == "S":
        raw = b"".join(v) if v.dtype.itemsize == 1 else v[0]
        return raw.decode("utf-8", "ignore").strip()
    return str(v[0]).strip()

def _strip_bytes_repr(df: pd.DataFrame) -> pd.DataFrame:
    """Turn leftover "b'text   '" strings (from an older CSV) into 'text'."""
    return df.apply(
        lambda col: col.str.replace(r"^b'(.*)'$", r"\1", regex=True).str.strip())

def _read_meta_files(dac: str) -> pd.DataFrame:
    """Read every *_meta.nc file for one DAC into a DataFrame (slow)."""
    rows = []
    for path in sorted(glob.glob(f"{GDAC}/{dac}/*/*_meta.nc")):
        with xr.open_dataset(path, decode_times=False) as ds:
            row = {"wmo": os.path.basename(path).split("_")[0]}
            row.update({f: get_str(ds, f) for f in FIELDS})
            rows.append(row)
    return pd.DataFrame(rows, columns=["wmo", *FIELDS])


@cache
def load_meta(dac: str = "aoml", refresh: bool = False) -> pd.DataFrame:
    """Float metadata for one DAC.

    Reads from the saved CSV if it exists; otherwise (or if refresh=True)
    rebuilds it from the netCDF files and saves the CSV.
    """
    csv_path = DATA_DIR / f"{dac}_float_meta.csv"
    if csv_path.exists() and not refresh:
        # dtype=str keeps WMO IDs as text; keep_default_na=False keeps '' as ''
        return pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    meta = _read_meta_files(dac)
    DATA_DIR.mkdir(exist_ok=True)
    meta.to_csv(csv_path, index=False)
    return meta


def get_wmos(meta: pd.DataFrame, pattern: str, cols: list[str]) -> set[str]:
    """WMO IDs where any of `cols` matches regex `pattern` (case-insensitive)."""
    hits = meta[cols].apply(
        lambda col: col.str.contains(pattern, case=False, na=False, regex=True))
    return set(meta.loc[hits.any(axis=1), "wmo"])

def get_sio_wmos(meta: pd.DataFrame) -> set[str]:
    return get_wmos(meta, SIO_PATTERN, INST_COLS)
 
 
def get_ffst_wmos(meta: pd.DataFrame) -> set[str]:
    return get_wmos(meta, FFST_PATTERN, PROJ_COLS)


if __name__ == "__main__":
    meta = load_meta()
    print(meta["OPERATING_INSTITUTION"].value_counts().head(20))
    print(meta["PROJECT_NAME"].value_counts().head(20))
    sio_wmos = get_sio_wmos(meta)
    ffst_wmos = get_ffst_wmos(meta)
    print(f"{len(sio_wmos)} SIO floats out of {len(meta)}")
    print(f"{len(ffst_wmos)} FFST floats out of {len(meta)}")