# src/sststack/plotting.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_FRIENDLY = {
    "GSST": "Global SST",
    "GMST": "Global SAT (scaled)",
    "GSST_NH": "SST — Northern Hemisphere",
    "GSST_TR": "SST — Tropics",
    "GSST_SH": "SST — Southern Hemisphere",
    "GSST_NA": "SST — North Atlantic",
    "GSST_SA": "SST — South Atlantic",
    "GSST_NP": "SST — North Pacific",
    "GSST_EP": "SST — Equatorial Pacific",
    "GSST_SP": "SST — South Pacific",
    "GSST_I":  "SST — Indian",
    "HT":      "Hemispheric gradient (NH − SH)",
}

def _load_pct_files(pct_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all *_pct.txt data to pandas DataFrames"""

    out: Dict[str, pd.DataFrame] = {}
    for f in sorted(pct_dir.glob("*_pct.txt")):
        key = f.name.replace("_pct.txt", "")
        df = pd.read_csv(f, delimiter="\t")

        # Security check
        cols = [f"pct_{i}" for i in range(1, 100)]
        df = df[[c for c in cols if c in df.columns]]
        out[key] = df
    return out

def _median(df: pd.DataFrame) -> np.ndarray:
    # Export median
    return df["pct_50"].to_numpy()

def _iter_bands(df: pd.DataFrame):
    """Loop to find the neighboor percentile around the median (fill_between). Weigth to modulate the alpha."""

    for k in range(1, 50):
        lo = df.get(f"pct_{50-k}", None)
        hi = df.get(f"pct_{50+k}", None)
        if lo is None or hi is None:
            continue
        ylo = lo.to_numpy()
        yhi = hi.to_numpy()

        weight = 1.0 - (k / 50.0)  # 0.98, ..., ~0.0
        yield ylo, yhi, weight

def _make_axes(nplots: int) -> Tuple[plt.Figure, List[plt.Axes]]:
    ncols = int(np.ceil(np.sqrt(nplots)))
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    return fig, [ax for row in axes for ax in row]

def plot_stacks_from_pct(
    pct_dir: Path,
    time: np.ndarray,
    outfile: Optional[Path] = None,
    zero_line: float = 0.0,
) -> Path:
    """
    Read the data files and build subplots with:
    - line => median (pct_50)
    - shaded envelope => (pct_49-51, pct_48-52, ...)
    - horizontal line => PI reference
    """
    pct_dir = Path(pct_dir)
    stacks = _load_pct_files(pct_dir)
    if not stacks:
        raise FileNotFoundError(f"No *_pct.txt found in {pct_dir}")

    fig, axes = _make_axes(len(stacks))
    for ax in axes[len(stacks):]:
        ax.set_visible(False)

    for ax, (key, df) in zip(axes, stacks.items()):
        # shaded envelope
        for ylo, yhi, w in _iter_bands(df):
            alpha = max(0.02, w**1.6 * 0.5)
            ax.fill_between(time, ylo, yhi, alpha=alpha, linewidth=0, color='c')

        # median
        ax.plot(time, _median(df), linewidth=2, color="black")

        # PI reference
        ax.axhline(zero_line, linewidth=0.8, linestyle="-")

        # axes
        ax.set_title(_FRIENDLY.get(key, key), fontsize=10)
        ax.set_xlabel("Age (ka)")
        ax.set_ylabel("ΔT (°C)")
        ax.grid(False)

    fig.tight_layout()
    outfile = outfile or (pct_dir / "stacks_summary.png")
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outfile

def plot_all(data_root: Path, version: str, time: np.ndarray, out: Optional[Path] = None) -> Path:
    """Convenience: calculates pct_dir from data_root/version and calls plot_stacks_from_pct."""
    pct_dir = Path(data_root) / "outputs" / "stacks" / version / "pct"
    out = out or (Path(data_root) / "outputs" / "stacks_summary.png")
    return plot_stacks_from_pct(pct_dir, time, outfile=out)
