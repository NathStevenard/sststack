import logging
from pathlib import Path

def setup_logger(run_dir: str, name: str = "sststack", level: int = logging.INFO) -> logging.Logger:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(run_dir) / "sststack.log"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler(); ch.setFormatter(fmt); ch.setLevel(level)
        fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt); fh.setLevel(level)
        logger.addHandler(ch); logger.addHandler(fh)
    return logger
