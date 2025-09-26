import argparse, time
from pathlib import Path
from sststack.io.logging_utils import setup_logger
from sststack.io.dataset import Dataset
from sststack.core.stacking import Stacker, TIME

def main():
    p = argparse.ArgumentParser(prog="sststack", description="SST stacking pipeline")
    p.add_argument("--data-root", required=True, help="Path folder 'examples/data'")
    p.add_argument("--version", default="v0.1")
    p.add_argument("--nsim", type=int, default=1000)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--filter", choices=["d18O","selected", None], default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / f"{timestamp}_{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(run_dir))

    logger.info("== sststack run ==")
    logger.info(f"params: nsim={args.nsim} blocks={args.blocks} filter={args.filter} version={args.version} seed={args.seed}")
    logger.info(f"data_root: {args.data_root}")

    ds = Dataset(Path(args.data_root), version=args.version, filter=args.filter)
    sst, age = ds.preload_files()
    info = ds.work_info()

    stacker = Stacker(
        sst=sst, age=age, info=info, nsim=args.nsim, blocks=args.blocks,
        version=args.version, output_root=str(Path(args.data_root).parent), seed=args.seed
    )
    stacker.run(logger)

    report = run_dir / "log_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# sststack run report\n\n")
        f.write(f"- nsim: {args.nsim}\n- blocks: {args.blocks}\n- filter: {args.filter}\n- version: {args.version}\n- seed: {args.seed}\n")
        f.write(f"- data_root: {args.data_root}\n- time steps: {len(TIME)} ({TIME[0]}–{TIME[-1]} ka, step 0.5)\n")
    logger.info(f"Report written: {report}")
    logger.info("Done.")
