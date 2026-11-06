"""Map recently reporting US Argo floats from the GDAC profile index."""
from functools import cache

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd

from argo_maps.metadata import GDAC, PROJECT_ROOT, load_meta, get_sio_wmos, get_ffst_wmos

FIG_DIR = PROJECT_ROOT / "figures"
CORE_CODES = [873, 875, 876, 877, 848, 878, 844, 837, 871, 870, 869, 861, 
              860, 857, 856, 858, 865, 863, 831, 843, 836, 835, 834, 842, 
              841, 839, 891, 890, 889, 888, 880, 854, 872, 852, 851, 853, 
              879, 847, 846, 882]
DEEP_CODES = [838, 849, 862, 864, 874, 883, 881]
BGC_CODES = [885, 884, 887, 886]


@cache
def load_profile_index() -> pd.DataFrame:
    """The GDAC global profile index, with dac, wmo, and cycle columns added."""
    prof = pd.read_csv(f"{GDAC}/ar_index_global_prof.txt", comment="#",
                       dtype={"file": str, "date": str, "profiler_type": "Int64"})
    prof["date"] = pd.to_datetime(prof["date"], format="%Y%m%d%H%M%S", errors="coerce")
    parts = prof["file"].str.split("/")
    prof["dac"] = parts.str[0]
    prof["wmo"] = parts.str[1]
    prof["cycle"] = prof["file"].str.extract(r"_(\d+)D?\.nc$")[0].astype("Int64")
    return prof


def latest_positions(prof: pd.DataFrame, since: pd.Timestamp) -> pd.DataFrame:
    """Newest profile with a valid position for each float, reported on or after `since`."""
    fixes = prof.dropna(subset=["latitude", "longitude", "date"])
    latest = fixes.sort_values("date").groupby("wmo").tail(1)
    return latest[latest["date"] >= since]


def plot_map(layers, title):
    """layers: list of (DataFrame, label, scatter-style dict), drawn in order."""
    fig = plt.figure(figsize=(13, 6.5))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=-160))
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.85")
    ax.coastlines(linewidth=0.4)
    for df, label, style in layers:
        ax.scatter(df["longitude"], df["latitude"], transform=ccrs.PlateCarree(),
                   label=f"{label} ({len(df)})", **style)
    ax.legend(loc="lower left")
    ax.set_title(title)
    return fig, ax

# US Argo Floats by Mission
if __name__ == "__main__":
    prof = load_profile_index()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    recent = latest_positions(prof, since=now - pd.Timedelta(days=30))

    us = recent[recent["dac"] == "aoml"]
    core = us[us["profiler_type"].isin(CORE_CODES)]
    deep = us[us["profiler_type"].isin(DEEP_CODES)]
    bgc = us[us["profiler_type"].isin(BGC_CODES)]

    fig, ax = plot_map(
        [
            (core, "US Core Argo",  dict(s=6, color="tab:blue")),
            (deep, "US Deep Argo",  dict(s=18, color="tab:red", marker="^")),
            (bgc, "US BGC Argo",  dict(s=18, color="tab:green", marker="^")),
        ],
        title=f"US Argo floats reporting in the 30 days to {now:%Y-%m-%d}",
    )
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "us_argo_mission_map.png", dpi=600, bbox_inches="tight")
    plt.show()

# SIO Argo Floats
if __name__ == "__main__":
    prof = load_profile_index()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    recent = latest_positions(prof, since=now - pd.Timedelta(days=30))

    us = recent[recent["dac"] == "aoml"]
    sio = us[us["wmo"].isin(get_sio_wmos(load_meta()))]

    fig, ax = plot_map(
        [
            (sio,  "SIO",           dict(s=10, color="tab:blue")),
        ],
        title=f"SIO Argo floats reporting in the 30 days to {now:%Y-%m-%d}",
    )
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "sio_argo_map.png", dpi=600, bbox_inches="tight")
    plt.show()

    # SIO Argo Floats by Mission
if __name__ == "__main__":
    prof = load_profile_index()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    recent = latest_positions(prof, since=now - pd.Timedelta(days=30))

    us = recent[recent["dac"] == "aoml"]
    sio = us[us["wmo"].isin(get_sio_wmos(load_meta()))]
    core = sio[sio["profiler_type"].isin(CORE_CODES)]
    deep = sio[sio["profiler_type"].isin(DEEP_CODES)]
    bgc = sio[sio["profiler_type"].isin(BGC_CODES)]

    fig, ax = plot_map(
        [
            (core, "SIO Core Argo",  dict(s=6, color="tab:blue")),
            (deep, "SIO Deep Argo",  dict(s=18, color="tab:red", marker="^")),
            (bgc, "SIO BGC Argo",  dict(s=18, color="tab:green", marker="^")),
        ],
        title=f"SIO Argo floats reporting in the 30 days to {now:%Y-%m-%d}",
    )
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "sio_argo_mission_map.png", dpi=600, bbox_inches="tight")
    plt.show()


    # FFST & Deep Argo Floats
if __name__ == "__main__":
    prof = load_profile_index()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    recent = latest_positions(prof, since=now - pd.Timedelta(days=30))

    us = recent[recent["dac"] == "aoml"]
    deep = us[us["profiler_type"].isin(DEEP_CODES)]
    ffst = us[us["wmo"].isin(get_ffst_wmos(load_meta()))]

    fig, ax = plot_map(
        [
            (deep, "US Deep Argo",  dict(s=10, color="tab:red")),
            (ffst, "FFST Deep Argo",  dict(s=18, color="tab:orange", marker="^")),
        ],
        title=f"US Deep Argo floats reporting in the 30 days to {now:%Y-%m-%d}",
    )
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "sio_argo_mission_map.png", dpi=600, bbox_inches="tight")
    plt.show()