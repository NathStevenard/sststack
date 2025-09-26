# sststack — Global and regional stacking SST-based Temperature Reconstructions

**"sststack"** is a Python package to build reproducible global and regional temperature stacks from **sea-surface temperature (SST) reconstructions**.  
The method follows the approach described in Stevenard et al. (2025)
---

## Installation

Clone the repository and install in *editable* mode:

```bash
git clone https://github.com/NathStevenard/sststack
cd sststack
python -m venv .venv
source .venv/bin/activate   # on Linux/Mac
# .venv\Scripts\activate    # on Windows PowerShell

pip install -e .[dev]
```

This will install:
- core dependencies (numpy, pandas, scipy, tqdm,...)
- development extras (pytest) if you use [dev]

After installation you should be able to run:
```bash
sststack --help
```
You can try to run the *test_import.py* to ensure the installation is done.

## Data formatting requirements

Your input data must follow the **exact same folder structure and naming convention** as in the example:
```markdown
data_root/ (your personnal folder)
└── <PERIOD>/
    ├── Core_information.xlsx
    ├── ens_age/
    │   └── {core}_{proxy}.txt
    └── ens_anomalies_annual/
        └── {core}_{proxy}.txt
```
- **Core_information.xlsx**
Must follow the template provided in *examples/data/MIS9/Core_information.xlsx*
Required columns: core, method, group, latitude, longitude
- **File naming convention**
Every file in ens_age/ must match exactly the corresponding file in ens_anomalies_annual/.
Example:
```
ens_age/ODP-1146_MgCa.txt
ens_anomalies_annual/ODP-1146_MgCa.txt
```
The {core} and {proxy} parts must be identical in the two repositories.

- **File format**
Each file must be a N x 1000 tab-delimited matrix (no header):
    - N = number of horizons (observed depths or ages)
    - 1000 = number of ensemble members
- **Annual only**
Only annual reconstructions are supported. Do not include seasonal files directory.
- **Consistency**
The set of files in ens_age/ and ens_anomalies_annual/ must match 1-to-1.

Note: if you have only a SST anomalies and standard deviation, please create a normalize random dataset with a N x 1000 size.

## Quick start with the template

The easiest way to use the package is via the provided template script:
```bash
python examples.example_MIS9.py
```
At the top of the template you can define your input settings:
```python
# examples/example_MIS9.py

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
nsim        = 200                               # (int)  || Number of simulations
blocks      = 10                                # (int)  || Number of parallel blocks (multiprocessing)
filter      = "selected"                        # (str)  || "d18O", "selected" or None
seed        = 42                                # (int)  || Global seed (None for random)
export_xlsx = True                              # (bool) || Write an excel file (summary)

# --- Time axis ----------------------------
TIME        = np.arange(300, 350.5, 0.5)        # format -> (start, end+step, step) <=== /!\ IMPORTANT /!\

# --- Ban-list -----------------------------
# Write the filenames you want to exclude from the process (if filter == "selected").
BAN_LIST    = {'ODP-1146_d18O.txt','MD06-3074B_d18O.txt',... }
```
Then just run the script, it will:
- load your data and metadata,
- run the stacking with parallel blocks (multiprocessing),
- save ensemble (ens/*.txt) and percentile (pct/*.txt) outputs,
- write logs and a report in runs/<timestamp>_<version>/,
- optionally export an Excel workbook,
- and generate summary figures with the function plot_all().

## Outputs

After running, you will find:
- **Stack results**
```markdown
examples/outputs/stacks/<version>/
├── ens/                        # raw ensemble members (one column per simulation)
├── pct/                        # percentile summaries (pct_1 ... pct_99)
├── temporary/                  # per-block intermediate results
└── stacks_<version>.xlsx       # final stacks results (age, pct_ 2, pct_16, pct_50, pct_84, pct_98) 
```
- **Run logs**
```markdown
examples/runs/<timestamp>_<version>/
├── sststack.log    # Run logs
└── log_report.md   # Run settings
```
- **Figure**
```markdown
examples/outputs/stacks_summary.png
```

## Example figure

The plotting utility (sststack.plotting.plot_all) automatically builds multi-panels figures:
- **black line** = median (50th percentile),
- **shaded envelop** = percentile bands (49-51, 48-52,...) with decreasing opacity,
- **horizontal line** = the PI reference (0°C).

## Citations

If you use this package in your research, please cite:
- The original article: (full citation incoming)
- If still in preprint, cite: 
```
Stevenard, N., Capron, É., Legrain, É., and Coutelle, C.: Global and regional sea-surface temperature changes over the Marine Isotopic Stage 9e and Termination IV, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2025-1928, 2025.
```
- This repository: https://github.com/NathStevenard/sststack.git

## License

MIT License - free to use, modify, and share, with attribution.