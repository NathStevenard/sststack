import argparse, time as _time
from pathlib import Path
import numpy as np

from ..io.logging_utils import setup_logger
from ..io.dataset import Dataset
from ..core.stacking import Stacker

def main():
    p = argparse.ArgumentParser(prog="sststack", description="SST stacking pipeline")
    p.add_argument("--data-root", required=True, help="Folder containing <period>/")
    p.add_argument("--period", default="MIS9", help="Subfolder name under data-root")
    p.add_argument("--version", default="v0.1")
    p.add_argument("--nsim", type=int, default=1000)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--filter", choices=["d18O", "selected", None], default=None)
    p.add_argument("--seed", type=int, default=42)

    # Optional controls for the time axis (with sensible defaults)
    p.add_argument("--time-start", type=float, default=300.0)
    p.add_argument("--time-end",   type=float, default=350.5)
    p.add_argument("--time-step",  type=float, default=0.5)

    # (Optional) write Excel summary
    p.add_argument("--export-xlsx", action="store_true")

    args = p.parse_args()

    # Build time axis locally (no reliance on stacking.py constants)
    time_axis = np.arange(args.time_start, args.time_end + args.time_step, args.time_step)

    timestamp = _time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / f"{timestamp}_{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(run_dir))

    logger.info("== sststack run ==")
    logger.info(
        f"params: nsim={args.nsim} blocks={args.blocks} filter={args.filter} "
        f"version={args.version} seed={args.seed}"
    )
    logger.info(
        f"period={args.period} data_root={args.data_root} "
        f"time=[{time_axis[0]}..{time_axis[-1]} step {time_axis[1]-time_axis[0]}]"
    )

    ds = Dataset(Path(args.data_root), version=args.version, filter=args.filter, period=args.period)
    sst, age = ds.preload_files()
    info = ds.work_info()

    stacker = Stacker(
        sst=sst,
        age=age,
        info=info,
        nsim=args.nsim,
        blocks=args.blocks,
        version=args.version,
        output_root=str(Path(args.data_root).parent),
        time=time_axis,
        period=args.period,
        seed=args.seed,
    )
    stacker.run(logger)

    if args.export_xlsx and hasattr(ds, "export_stacks_xlsx"):
        try:
            xlsx = ds.export_stacks_xlsx(time=time_axis)
            logger.info(f"Excel summary: {xlsx}")
        except Exception as exc:
            logger.exception(f"Excel export failed: {exc}")

    # Write a run report (nice for reproducibility)
    report = run_dir / "log_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# sststack run report\n\n")
        f.write(f"- period: {args.period}\n- version: {args.version}\n")
        f.write(f"- nsim: {args.nsim}\n- blocks: {args.blocks}\n")
        f.write(f"- filter: {args.filter}\n- seed: {args.seed}\n")
        f.write(f"- data_root: {args.data_root}\n")
        f.write(f"- time steps: {len(time_axis)} "
                f"({time_axis[0]}–{time_axis[-1]} ka, step {time_axis[1]-time_axis[0]})\n")
    logger.info(f"Report written: {report}")
