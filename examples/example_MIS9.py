"""
example_MIS9.py — Example of stacking with the MIS 9 annual SST data (see Stevenard et al., 2025)

Modify the settings below and run the script.
"""

from __future__ import annotations
from pathlib import Path
import time
import numpy as np

# ========================
# ==  INPUTS — TO EDIT  ==
# ========================

# --- Basic settings -----------------------
data_root   = Path(__file__).parent / "data"    # (path) || Folder containig the data folder (here it's "MIS9")
period      = "MIS9"                            # (str)  || Name of the data folder <=== /!\ IMPORTANT /!\
version     = "v0.1-demo"                       # (str)  || Output tag
nsim        = 100                               # (int)  || Number of simulations
blocks      = 10                                # (int)  || Number of parallel blocks (multiprocessing)
filter      = "selected"                        # (str)  || "d18O", "selected" or None
seed        = 42                                # (int)  || Global seed (None for random)
export_xlsx = True                              # (bool) || Write an excel file (summary)

# --- Time axis ----------------------------
TIME        = np.arange(300, 350.5, 0.5)        # format -> (start, end+step, step) <=== /!\ IMPORTANT /!\

# --- Ban-list -----------------------------
# Write the filenames you want to exclude from the process (if filter == "selected").
BAN_LIST    = {
    'ODP-1146_d18O.txt','MD06-3074B_d18O.txt','MD03-2699_d18O.txt','MD97-2140_d18O.txt',
    'MD01-2416_d18O.txt','ODP-806_d18O.txt','MD02-2575_d18O.txt','ODP-999_MgCa.txt',
    'MD96-2080_MgCa.txt','MD06-2986_MgCa.txt','ODP-1172_d18O.txt','U1446_MgCa.txt','V19-30_d18O.txt'
}

# ========================
# == CODE — DO NOT MODIFY
# ========================

# Compatible import (depending on the project structure)
try:
    from sststack.io.dataset import Dataset
    from sststack.core.stacking import Stacker
    from sststack.io.logging_utils import setup_logger
    from sststack.plotting import plot_all
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

    from plotting import plot_all # type: ignore
    from dataset import Dataset  # type: ignore
    from stacking import Stacker  # type: ignore

# ======================================================================

def main() -> None:

    # --- Setup logs -------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / f"{timestamp}_{version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(run_dir))
    logger.info("== sststack MIS9 example run ==") # <-- YOU CAN CHANGE THE STRING HERE
    logger.info(
        f"params: nsim={nsim}, blocks={blocks}, filter={filter}, version={version}, seed={seed}")
    logger.info(f"data_root: {data_root}")

    # --- Load data -------------
    ds = Dataset(
        Path(data_root),
        version=version,
        filter=filter,
        period=period,
        ban_list=BAN_LIST
    )
    sst, age = ds.preload_files()
    info = ds.work_info()

    # --- Stacking process ------
    stacker = Stacker(
        sst=sst,
        age=age,
        info=info,
        nsim=nsim,
        blocks=blocks,
        version=version,
        output_root=str(Path(data_root).parent),
        time=TIME,
        period=period,
        seed=seed
    )
    stacker.run(logger)

    # --- Excel export (optionnal)

    if export_xlsx and hasattr(ds, "export_stacks_xlsx"):
        try:
            pct_dir = Path(__file__).parent / "outputs" / "stacks" / version / "pct"
            xlsx = ds.export_stacks_xlsx(time=TIME, pct_dir=pct_dir)
            logger.info(f"Excel report ==> OK: {xlsx}")
        except Exception as exc:
            logger.exception(f"[WARNING] Excel export failed: {exc}")

    # --- Export log report
    report = run_dir / "log_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# sststack run report\n\n")
        f.write(f"- period: {period}\n- version: {version}\n- nsim: {nsim}\n- blocks: {blocks}\n")
        f.write(f"- filter: {filter}\n- seed: {seed}\n- data_root: {data_root}\n")
        f.write(f"- time steps: {len(TIME)} ({TIME[0]}–{TIME[-1]} ka, step {TIME[1]-TIME[0]})\n")
    logger.info(f"Report written: {report}")

    # --- Summary --------------
    print("Stacks exported in: ", Path(data_root) / period / "stacks" / version)
    print("Time (ka): ", TIME[0], "to", TIME[-1], " step of ", TIME[1]-TIME[0], " ka")

if __name__ == "__main__":
    main()

# --- Plot the results ----------------------------------------------------
png_path = plot_all(data_root=Path(__file__).parent, version=version, time=TIME)
print(f"Summary figure saved at: {png_path}")
