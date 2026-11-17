"""Map recently reporting US Argo floats from the GDAC profile index."""
from functools import cache

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from cartopy.mpl.ticker import LatitudeFormatter

from argo_maps.metadata import GDAC, PROJECT_ROOT, load_meta, get_sio_wmos, get_ffst_wmos

FIG_DIR = PROJECT_ROOT / "figures"
CORE_CODES = [873, 875, 876, 877, 848, 878, 844, 837, 871, 870, 869, 861, 
              860, 857, 856, 858, 865, 863, 831, 843, 836, 835, 834, 842, 
              841, 839, 891, 890, 889, 888, 880, 854, 872, 852, 851, 853, 
              879, 847, 846, 882]
DEEP_CODES = [838, 849, 862, 864, 874, 883, 881]
BGC_CODES = [885, 884, 887, 886]

# Placeholder until there is a get_pga_wmos() in argo_maps.metadata
PGA_WMOS: list[str] = []

# set colors
CORE_STYLE = dict(s=5, color="#1f77b4", alpha=0.65, linewidths=0, zorder=3)
BGC_STYLE = dict(s=22, color="#2ca02c", marker="^", edgecolors="white",
                 linewidths=0.4, zorder=4)
DEEP_STYLE = dict(s=22, color="#d62728", marker="^", edgecolors="white",
                  linewidths=0.4, zorder=5)


# FFST figure: every deep float is a small red dot; subsets are rings on top
DEEP_BASE_STYLE = dict(s=10, color="#d62728", alpha=0.9, linewidths=0, zorder=3)
FFST_STYLE = dict(s=55, marker="o", facecolors="none", edgecolors="#1a1a1a",
                  linewidths=0.9, zorder=5)
PGA_STYLE = dict(s=60, marker="D", facecolors="none", edgecolors="#7b2d8e",
                 linewidths=0.9, zorder=4)

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


def plot_map(layers, date: pd.Timestamp, title: str | None = None):
    """layers: list of (DataFrame, label, scatter-style dict), drawn in order.
    
    N below the map counts unique floats across all layers, so floats that
    appear in more than one (non-exclusive) layer are counted once.
    """

    date_str = f"{date:%d-%b-%Y}"
    if title is None:
        title = f"Argo Fleet: active floats as of {date_str}"

    fig = plt.figure(figsize=(13, 6.5))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=-160))
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=1)
    ax.coastlines(linewidth=0.4, zorder=2)

    # Dashed 30-degree graticule; latitude labels only (60S-60N)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        xlocs=mticker.FixedLocator(range(-180, 181, 30)),
        ylocs=mticker.FixedLocator([-60, -30, 0, 30, 60]),
        linestyle="--", linewidth=0.5, color="0.5", alpha=0.7, zorder=2,
    )
    gl.top_labels = gl.bottom_labels = gl.right_labels = False
    gl.xlines = True
    gl.yformatter = LatitudeFormatter()
    gl.ylabel_style = dict(size=9, color="0.3")

    for df, label, style in layers:
        ax.scatter(df["longitude"], df["latitude"], transform=ccrs.PlateCarree(),
                   label=f"{label} ({len(df)})", **style)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title(title)

    n_total = pd.concat([df["wmo"] for df, _, _ in layers]).nunique()
    ax.text(0.5, -0.03, f"(N={n_total})", transform=ax.transAxes,
            ha="center", va="top", fontsize=11)
    ax.text(1.0, -0.03, f"U.S. Argo Program {date_str}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="0.35")

    return fig, ax

def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=600, bbox_inches="tight")

if __name__ == "__main__":
    prof = load_profile_index()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    recent = latest_positions(prof, since=now - pd.Timedelta(days=30))
    us = recent[recent["dac"] == "aoml"]
    meta = load_meta()


# US Argo Floats by Mission
    core = us[us["profiler_type"].isin(CORE_CODES)]
    deep = us[us["profiler_type"].isin(DEEP_CODES)]
    bgc = us[us["profiler_type"].isin(BGC_CODES)]

    fig, ax = plot_map(
        [
            (core, "U.S. Core Argo",  CORE_STYLE),
            (bgc, "U.S. BGC Argo",  BGC_STYLE),
            (deep, "U.S. Deep Argo",  DEEP_STYLE),
        ],
        date=now,
    )
    save(fig, "us_argo_mission_map.png")

# SIO Argo Floats
    sio = us[us["wmo"].isin(get_sio_wmos(load_meta()))]

    fig, ax = plot_map(
        [
            (sio,  "SIO", dict(CORE_STYLE, s=8))
        ],
        date=now
    )
    save(fig, "sio_argo_map.png")

    # SIO Argo Floats by Mission
    sio_core = sio[sio["profiler_type"].isin(CORE_CODES)]
    sio_bgc = sio[sio["profiler_type"].isin(BGC_CODES)]
    sio_deep = sio[sio["profiler_type"].isin(DEEP_CODES)]

    fig, ax = plot_map(
        [
            (sio_core, "SIO Core Argo", CORE_STYLE),
            (sio_bgc, "SIO BGC Argo", BGC_STYLE),
            (sio_deep, "SIO Deep Argo", DEEP_STYLE),
            
        ],
        date=now,
    )
    save(fig, "sio_argo_mission_map.png")


    # US Deep Argo with FFST and PGA highlighted (subsets, not exclusive)
    ffst = us[us["wmo"].isin(get_ffst_wmos(load_meta()))]
    pga = us[us["wmo"].isin(PGA_WMOS)]

    fig, ax = plot_map(
        [
            (deep, "U.S. Deep Argo", DEEP_BASE_STYLE),   # base layer first
            (ffst, "FFST Deep Argo", FFST_STYLE),
            (pga, "PGA Deep Argo", PGA_STYLE),
        ],
        date=now,
    )
    save(fig, "us_argo_deep_ffst_map.png")


    plt.show()