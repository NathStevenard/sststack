import os, random, warnings
from dataclasses import dataclass
from typing import Dict
import numpy as np, pandas as pd
from scipy.spatial import Delaunay
from pandas.errors import PerformanceWarning

# -------------------- helpers ---------------------------------------------------------------
def _run_block_worker(args):
    (block_idx, sim_per_block, seed, stacker_state) = args
    # configure random seed for the block
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    # A mini-stacker is "reconnected" from the serializable state.
    self = stacker_state["self"]
    results = {k: [] for k in stacker_state["keys"]}
    for _ in range(sim_per_block):
        res = self._simulate_once()
        for k in results:
            results[k].append(res[k])
    return block_idx, results

@dataclass
class Stacker:
    sst: Dict[str, np.ndarray]
    age: Dict[str, np.ndarray]
    info: pd.DataFrame
    nsim: int
    blocks: int
    version: str
    output_root: str
    time: np.ndarray
    period: str = "MIS9"
    envlp: float = 0.25
    seed: int | None = 42

    def __post_init__(self):
        self.output_path = os.path.join(self.output_root, "outputs", "stacks", self.version)
        os.makedirs(self.output_path, exist_ok=True)
        for b in range(1, self.blocks+1):
            os.makedirs(os.path.join(self.output_path, f"temporary/block_{b}"), exist_ok=True)
        if self.seed is not None:
            random.seed(self.seed); np.random.seed(self.seed)

    # ---------- helpers ----------
    def _random_grid(self):
        r = random.uniform(2,5)
        lat_grid = np.array(list(np.arange(-60, 60, r)) + [60])
        lon_grid = np.array(list(np.arange(-180, 180, r)) + [180])
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        stations = self.info[['latitude','longitude']].values
        points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
        tri = Delaunay(points)
        idx = tri.find_simplex(stations)
        nearest = tri.simplices[idx]
        self.info['neigh_lat'] = [points[n[0], 0] for n in nearest]
        self.info['neigh_lon'] = [points[n[0], 1] for n in nearest]
        self.info['filename'] = self.info['filename'].apply(lambda f: f[0] if isinstance(f, list) else f)
        grid_grouped = self.info.groupby(['neigh_lat','neigh_lon']).agg({'filename': list, 'group':'first'})
        return grid_grouped

    def _grid_mean(self):
        grid_grouped = self._random_grid()
        results = []
        for (_, _), row in grid_grouped.iterrows():
            lst = []
            for file in row['filename']:
                data = self.sst[file].reshape((len(self.sst[file]) // 1000, 1000))
                age  = self.age[file].reshape((len(self.age[file])  // 1000, 1000))
                # shuffle age (preserve the N x 1000 structure)
                for i in range(age.shape[0]): np.random.shuffle(age[i])
                averages = []
                for t in self.time:
                    r_idx, c_idx = np.where((age >= t - self.envlp) & (age <= t + self.envlp))
                    averages.append(np.mean(data[r_idx, c_idx]) if len(r_idx) > 0 else np.nan)
                lst.append(averages)
            gdf = pd.DataFrame(lst, index=row['filename'], columns=self.time).T
            gdf['group_avg'] = gdf.mean(axis=1, skipna=True)
            results.append(gdf[['group_avg']])
        grid_sst = pd.concat(results, keys=[(lat,lon,grid_grouped['group'].loc[(lat,lon)])
                                            for lat,lon in grid_grouped.index])
        grid_sst.index.rename(['neigh_lat','neigh_lon','group','time'], inplace=True)
        return grid_sst

    @staticmethod
    def _assign_lat_band(lat, bands):
        for i in range(len(bands)-1):
            if bands[i] <= lat < bands[i+1]: return bands[i], bands[i+1]
        return bands[-2], bands[-1]

    def _lat_mean(self, grid_sst, bands):
        df = grid_sst.reset_index()
        df[['lat_mini','lat_maxi']] = df['neigh_lat'].apply(lambda x: self._assign_lat_band(x, bands)).apply(pd.Series)
        lat_band_avg = (df.groupby(['lat_mini','lat_maxi','time'])['group_avg'].mean().reset_index())
        return lat_band_avg.groupby(['lat_mini', 'lat_maxi']).agg({'time': list, 'group_avg': list}).reset_index()

    @staticmethod
    def _process_lat_band(list_of_lat, weighted=True):
        expanded = pd.DataFrame({
            'time': list_of_lat['time'].explode().reset_index(drop=True),
            'group_avg': [x for sub in list_of_lat['group_avg'] for x in sub]
        })
        expanded['lat_mini'] = list_of_lat['lat_mini'].repeat(list_of_lat['time'].apply(len)).reset_index(drop=True)
        expanded['lat_maxi'] = list_of_lat['lat_maxi'].repeat(list_of_lat['time'].apply(len)).reset_index(drop=True)
        if weighted:
            lat_min = pd.to_numeric(expanded['lat_mini'], errors='coerce')
            lat_max = pd.to_numeric(expanded['lat_maxi'], errors='coerce')
            expanded['weight'] = np.abs(np.sin(np.radians(lat_max)) - np.sin(np.radians(lat_min)))
        else:
            expanded['weight'] = 1.0
        expanded = expanded.dropna(subset=['group_avg','weight'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerformanceWarning)
            mean_series = (
                expanded.groupby("time")[["group_avg", "weight"]]
                .apply(lambda g: np.average(g["group_avg"], weights=g["weight"]))
            )
        return mean_series.tolist()

    # ---------- simulation ----------
    def _simulate_once(self):
        r_band = random.uniform(2.5,10)
        bands = np.arange(-60, 60 + r_band, r_band)
        sst_to_sat = random.uniform(1.5, 2.3)

        grid_sst = self._grid_mean()
        lat_all  = self._lat_mean(grid_sst, bands)
        global_sst = self._process_lat_band(lat_all)
        global_sat = [v * sst_to_sat for v in global_sst]

        def split_lat(df):
            idx = df.index.get_level_values('neigh_lat')
            return df[idx > 23], df[(idx > -23) & (idx < 23)], df[idx < -23]
        NH, TR, SH = split_lat(grid_sst)
        sNH = self._process_lat_band(self._lat_mean(NH, bands))
        sTR = self._process_lat_band(self._lat_mean(TR, bands))
        sSH = self._process_lat_band(self._lat_mean(SH, bands))

        def sel(df, g): return df[df.index.get_level_values('group') == g]
        basins = {k: sel(grid_sst,k) for k in ['North Atlantic','South Atlantic','North Pacific','Equatorial Pacific','South Pacific','Indian']}
        bproc = {k: self._process_lat_band(self._lat_mean(v, bands)) for k,v in basins.items()}

        return {'GSST': global_sst, 'GMST': global_sat,'GSST_NH': sNH, 'GSST_TR': sTR, 'GSST_SH': sSH,
                'GSST_NA': bproc['North Atlantic'], 'GSST_SA': bproc['South Atlantic'],
                'GSST_NP': bproc['North Pacific'], 'GSST_EP': bproc['Equatorial Pacific'],
                'GSST_SP': bproc['South Pacific'], 'GSST_I': bproc['Indian']}

    def run(self, logger):
        import concurrent.futures, time
        t0 = time.time()
        keys = ['GSST', 'GMST', 'GSST_NH', 'GSST_TR', 'GSST_SH', 'GSST_NA', 'GSST_SA', 'GSST_NP', 'GSST_EP', 'GSST_SP',
                'GSST_I']
        sim_per_block = int(self.nsim / self.blocks)

        # Prepare the temporary output folders
        for b in range(1, self.blocks + 1):
            bdir = os.path.join(self.output_path, f"temporary/block_{b}")
            os.makedirs(bdir, exist_ok=True)

        # Serializable state passed to workers
        stacker_state = {"self": self, "keys": keys}

        # Change random seeds for each block
        def seed_for_block(b):
            return None if self.seed is None else (self.seed * 1000 + b)

        # Parallel launch
        logger.info(f"Parallel run: {self.blocks} blocks × {sim_per_block} sims (total {self.nsim})")
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as ex:
            for b in range(1, self.blocks + 1):
                args = (b, sim_per_block, seed_for_block(b), stacker_state)
                futures.append(ex.submit(_run_block_worker, args))

            for fut in concurrent.futures.as_completed(futures):
                try:
                    block_idx, results = fut.result()
                except Exception as e:
                    logger.exception("Block crashed", exc_info=e)
                    continue

                # Export block ensembles
                bdir = os.path.join(self.output_path, f"temporary/block_{block_idx}")
                for k in keys:
                    df = pd.DataFrame(results[k]).T
                    df.columns = [f"sim_{(block_idx - 1) * sim_per_block + i + 1}" for i in range(sim_per_block)]
                    df.to_csv(os.path.join(bdir, f"{k}_ens.csv"))
                logger.info(f"Saved block {block_idx} ensembles to: {bdir}")

        logger.info(f"Simulation finished in {time.time() - t0:.1f}s")
        self.finalize(logger)

    def finalize(self, logger):
        keys = ['GSST','GMST','GSST_NH','GSST_TR','GSST_SH','GSST_NA','GSST_SA','GSST_NP','GSST_EP','GSST_SP','GSST_I']
        combined = {k: pd.DataFrame() for k in keys}
        for b in range(1, self.blocks+1):
            bdir = os.path.join(self.output_path, f"temporary/block_{b}")
            for k in keys:
                df = pd.read_csv(os.path.join(bdir, f"{k}_ens.csv"), index_col=0)
                combined[k] = pd.concat([combined[k], df], axis=1)

        # HT = NH - SH (Hemispheric Heat Transfer)
        NH_sorted = combined['GSST_NH'].apply(lambda row: pd.Series(sorted(row.values)), axis=1)
        SH_sorted = combined['GSST_SH'].apply(lambda row: pd.Series(sorted(row.values)), axis=1)
        HT = NH_sorted - SH_sorted

        ens_dir = os.path.join(self.output_path, 'ens')
        pct_dir = os.path.join(self.output_path, 'pct')
        os.makedirs(ens_dir, exist_ok=True); os.makedirs(pct_dir, exist_ok=True)

        for k,v in combined.items():
            v.to_csv(os.path.join(ens_dir, f"{k}_ens.txt"), sep='\t', index=False)
        HT.to_csv(os.path.join(ens_dir, 'HT_ens.txt'), sep='\t', index=False)

        percentiles = np.arange(1,100)
        def to_pct(df):
            return pd.DataFrame([np.percentile(df.iloc[i,:], percentiles) for i in range(df.shape[0])],
                                columns=[f"pct_{i+1}" for i in range(99)])
        for k,v in combined.items():
            to_pct(v).to_csv(os.path.join(pct_dir, f"{k}_pct.txt"), sep='\t', index=False)
        to_pct(HT).to_csv(os.path.join(pct_dir, "HT_pct.txt"), sep='\t', index=False)

        logger.info("Final export complete")
