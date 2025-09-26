from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, Iterable
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

# "selected" = list of files to exclude of the stacking process
DEFAULT_BAN_LIST = {
    'ODP-1146_d18O.txt','MD06-3074B_d18O.txt','MD03-2699_d18O.txt','MD97-2140_d18O.txt',
    'MD01-2416_d18O.txt','ODP-806_d18O.txt','MD02-2575_d18O.txt','ODP-999_MgCa.txt',
    'MD96-2080_MgCa.txt','MD06-2986_MgCa.txt','ODP-1172_d18O.txt','U1446_MgCa.txt','V19-30_d18O.txt'
}

def _safe_sheet_name(name: str) -> str:
    # Excel sheet name: max 31 chars, no []:*?/\
    s = re.sub(r'[:\\/*?\\[\\]]', '-', name)
    return s[:31]

@dataclass
class Dataset:
    data_root: Path
    version: str
    filter: Optional[str] = None                # None | "d18O" | "selected"
    period: str = "MIS9"
    ban_list: Optional[Iterable[str]] = None

    def __post_init__(self):
        base = Path(self.data_root) / self.period
        print("base is: ", base)

        self.path_sst = base / "ens_anomalies_annual"
        self.path_age = base / "ens_age"
        self.info_path = base / "Core_information.xlsx"
        self.output = self.data_root.parent / "outputs" / "stacks" / self.version
        print("self output is: ", self.output)
        for p in [self.output / "ens", self.output / "pct", self.output / "temporary"]:
            p.mkdir(parents=True, exist_ok=True)

        if not self.info_path.exists():
            raise FileNotFoundError(f"Missing metadata Excel: {self.info_path}")

        self.info = pd.read_excel(self.info_path)
        self.file_txt = [f for f in os.listdir(self.path_sst) if f.endswith(".txt")]
        self._ban = set(self.ban_list) if self.ban_list is not None else DEFAULT_BAN_LIST

    def _filtered_files(self) -> list[str]:
        out = []
        for f in self.file_txt:
            if self.filter == "d18O" and (f.endswith("d18O.txt") or f.endswith("d18Op.txt")):
                continue
            if self.filter == "selected" and f in self._ban:
                continue
            out.append(f)
        return out

    def preload_files(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        sst, age = {}, {}
        for fname in tqdm(self._filtered_files(), desc="Loading SST & age", unit="file"):
            sst[fname] = pd.read_csv(self.path_sst / fname, delimiter="\t", header=None).values.flatten()
            age[fname] = pd.read_csv(self.path_age / fname, delimiter="\t", header=None).values.flatten()
        return sst, age

    def work_info(self) -> pd.DataFrame:
        files = self._filtered_files()
        df = pd.DataFrame({
            "core":   [n.split("_")[0] for n in files],
            "method": [n.split("_")[1].split(".")[0] for n in files],
            "filename": files,
        })
        return df.merge(self.info[['core','method','group','latitude','longitude']],
                        on=['core','method'], how='left')

    def export_stacks_xlsx(self, time: np.ndarray, percentiles=(2, 16, 50, 84, 98), pct_dir: Path | None = None) -> Path:
        pct_dir = Path(pct_dir) if pct_dir is not None else (self.output / "pct")
        files = sorted(pct_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No percentile files found in {pct_dir}")

        # output path
        xlsx = pct_dir.parent / f"stacks_{self.version}.xlsx"
        xlsx.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(xlsx) as writer:
            for f in files:
                df = pd.read_csv(f, delimiter="\t")
                cols = [f"pct_{p}" for p in percentiles if f"pct_{p}" in df.columns]
                out = pd.DataFrame({"Age (ka)": time})
                out = pd.concat([out, df[cols].reset_index(drop=True)], axis=1)
                sheet = _safe_sheet_name(f.name.replace("_pct.txt", ""))
                out.to_excel(writer, sheet_name=sheet, index=False)
        return xlsx