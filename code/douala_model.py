#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial agent-based model of mobility, shared WASH use and cholera exposure.

The model defines urban spatial inputs, synthetic destinations, agent routines,
contamination dynamics, scenario configurations and ensemble analysis outputs.
Use ``ensemble_runner.py`` with the provided Slurm scripts for batch execution.
"""
import json

import matplotlib
matplotlib.use("Agg")  # headless / batch safe


# ===================== FINAL PAPER EPIDEMIOLOGY SETTINGS =====================
# These values are used consistently for baseline, intervention plots,
# seed-variability runs, grid sensitivity, and Sobol sensitivity.
# They are defined near the top because some early notebook cells seed
# tap/home contamination before the later R0b run-configuration block.
INITIAL_EXPOSED_FRAC = 0.01
INITIAL_INFECTED_FRAC = 0.02
TAP_INITIAL_CONTAMINATION_FRAC = 0.10
HOME_INITIAL_CONTAMINATION_FRAC = 0.012
TAP_EXPOSURE_PROB = 0.15
HOUSEHOLD_SECONDARY_EXPOSURE_PROB = 0.08

FINAL_EPI_PARAMS = {
    "INITIAL_EXPOSED_FRAC": INITIAL_EXPOSED_FRAC,
    "INITIAL_INFECTED_FRAC": INITIAL_INFECTED_FRAC,
    "TAP_INITIAL_CONTAMINATION_FRAC": TAP_INITIAL_CONTAMINATION_FRAC,
    "HOME_INITIAL_CONTAMINATION_FRAC": HOME_INITIAL_CONTAMINATION_FRAC,
    "TAP_EXPOSURE_PROB": TAP_EXPOSURE_PROB,
    "HOUSEHOLD_SECONDARY_EXPOSURE_PROB": HOUSEHOLD_SECONDARY_EXPOSURE_PROB,
}



# ===================== Notebook cell 5 =====================
# ── PATHS ──────────────────────────────────────────────────────────
import argparse
import csv
from pathlib import Path
import os
def pcca_cache_path(out_dir: str | Path, city_slug: str) -> Path:
    out_dir = Path(out_dir)
    return out_dir / city_slug / "precompute" / "pcca_best_thr.json"

def load_cached_pcca(out_dir: str | Path, city_slug: str):
    p = pcca_cache_path(out_dir, city_slug)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return None

def save_cached_pcca(out_dir: str | Path, city_slug: str, payload: dict):
    p = pcca_cache_path(out_dir, city_slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--city_slug", required=True, help="Matches city_slug in configs/cities.csv")
    p.add_argument("--run_id", type=int, default=0, help="Run number for CI / repetition")
    p.add_argument("--max_radius_m", type=int, default=20000, help="Network radius in meters")
    return p.parse_args()

def load_city_info(project_root: Path, city_slug: str):
    cities_path = project_root / "configs" / "cities.csv"
    assert cities_path.exists(), f"Missing {cities_path}"
    with open(cities_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["city_slug"].strip() == city_slug:
                place_name = row["place_name"].strip()
                lat = float(row["lat"])
                lon = float(row["lon"])
                return place_name, (lat, lon)
    raise ValueError(f"city_slug '{city_slug}' not found in {cities_path}")

ARGS = parse_args()

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../cholera_project
CITY_SLUG = ARGS.city_slug
RUN_ID = ARGS.run_id
MAX_RADIUS_M = ARGS.max_radius_m

# Inputs for this city
DATA_DIR = PROJECT_ROOT / "inputs" / CITY_SLUG / "data"
POP_RASTER = DATA_DIR / "pop.tif"
POP_XYZ = DATA_DIR / "pop.csv"

for p in [POP_RASTER, POP_XYZ]:
    assert p.exists(), f"Missing file: {p}"

# Outputs for this city + run
OUT_DIR = PROJECT_ROOT / "outputs" / CITY_SLUG / f"run_{RUN_ID:03d}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Make cache per-city (centrality cache should be re-used across runs)
CACHE_DIR = PROJECT_ROOT / "cache" / CITY_SLUG
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("AIRE_CENTRALITY_CACHE_DIR", str(CACHE_DIR))

# Load the city details (place name + center coordinates)
PLACE_NAME, CENTER_LATLON = load_city_info(PROJECT_ROOT, CITY_SLUG)

print(f"[CONFIG] CITY_SLUG={CITY_SLUG} RUN_ID={RUN_ID}")
print(f"[CONFIG] PLACE_NAME={PLACE_NAME}")
print(f"[CONFIG] CENTER_LATLON={CENTER_LATLON}")
print(f"[CONFIG] DATA_DIR={DATA_DIR}")
print(f"[CONFIG] OUT_DIR={OUT_DIR}")


# ===================== Notebook cell 6 =====================
import rasterio
import matplotlib.pyplot as plt
#──────────────────────────────────────────────────────────────────────────────
# PERFORMANCE HELPERS (Aire)
# - Optional: use NetworKit for multi-threaded exact betweenness (much faster).
# - Cache centrality results to disk so reruns don't recompute.
#   Set AIRE_CENTRALITY_CACHE_DIR to change location (default: ./cache).
#   Set AIRE_USE_NETWORKIT=1 to prefer NetworKit if installed.
#   Set AIRE_FORCE_RECOMPUTE=1 to ignore cache.
# ──────────────────────────────────────────────────────────────────────────────
import os, pickle, hashlib

def _centrality_cache_dir() -> Path:
    d = Path(os.environ.get("AIRE_CENTRALITY_CACHE_DIR", "cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d

def _graph_fingerprint(G, weight: str = "length") -> str:
    # Stable, cheap fingerprint based on node/edge counts + weighted edge stats
    n = G.number_of_nodes()
    m = G.number_of_edges()
    w_min = None
    w_max = None
    w_sum = 0.0
    w_cnt = 0
    for _, _, data in G.edges(data=True):
        w = data.get(weight, 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        w_cnt += 1
        w_sum += w
        w_min = w if w_min is None else min(w_min, w)
        w_max = w if w_max is None else max(w_max, w)
    payload = f"{n}|{m}|{w_cnt}|{w_sum:.6f}|{(w_min or 0):.6f}|{(w_max or 0):.6f}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]

def betweenness_centrality_fast(G_und, weight: str = "length", normalized: bool = True, seed: int = 0):
    fp = _graph_fingerprint(G_und, weight=weight)
    cache_path = _centrality_cache_dir() / f"betweenness_{fp}.pkl"
    if not os.environ.get("AIRE_FORCE_RECOMPUTE") and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    use_nk = os.environ.get("AIRE_USE_NETWORKIT", "0") == "1"
    if use_nk:
        try:
            import networkit as nk  # optional dependency
            # Convert to NetworKit graph (undirected) with edge weights
            # NetworKit expects node ids 0..n-1
            nodes = list(G_und.nodes())
            idmap = {u:i for i,u in enumerate(nodes)}
            Gnk = nk.graph.Graph(len(nodes), weighted=True, directed=False)
            for u, v, data in G_und.edges(data=True):
                w = data.get(weight, 1.0)
                try: w = float(w)
                except Exception: w = 1.0
                Gnk.addEdge(idmap[u], idmap[v], w)
            bc = nk.centrality.Betweenness(Gnk, normalized=normalized, computeEdgeCentrality=False)
            bc.run()
            scores = bc.scores()
            out = {nodes[i]: float(scores[i]) for i in range(len(nodes))}
            with open(cache_path, "wb") as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            return out
        except Exception as e:
            print("[centrality] NetworKit unavailable/failed, falling back to NetworkX:", e)

    # NetworkX fallback (single-threaded). Keep your existing behaviour:
    # exact for small graphs; sampled k for large graphs.
    n_nodes = G_und.number_of_nodes()
    if n_nodes <= 2000:
        out = nx.betweenness_centrality(G_und, weight=weight, normalized=normalized)
    else:
        k = min(1000, max(200, int(0.25 * n_nodes)))
        out = nx.betweenness_centrality(G_und, weight=weight, normalized=normalized, k=k, seed=seed)

    with open(cache_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out

with rasterio.open(POP_RASTER) as src:
    arr = src.read(1)
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, extent=extent, cmap="viridis", origin="upper")
    plt.title("Population Density (2020, persons/km²)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label="Population per km²")
   # plt.show()

# ===================== Notebook cell 7 =====================
# === CELL 8 — CITY CONFIG + POPULATION PCCA (fast) + COAST TRIM ===============

# 1. MANUAL CITY INPUTS --------------------------------------------------------
#PLACE_NAME     = "Addis Ababa, Arada, Addis Ababa, 3578, Ethiopia"
#CENTER_LATLON  = (9.02497, 38.74689)    # (lat, lon)
#MAX_RADIUS_M   = 20_000                  # search radius for PCCA & street network (m)

# 2. PCCA MODE SELECTION -------------------------------------------------------
#   "country" = Option A  → global/country-level threshold, then local boundary
#   "local"   = Option B  → fully local threshold around CENTER_LATLON (slower)
PCCA_MODE = "country"   # "country" or "local"

import time
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import networkx as nx
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import transform as _reproj_pts
from rasterio import features
from shapely.geometry import Point, box, Polygon, MultiPolygon
from shapely.ops import unary_union, polygonize, linemerge, snap
from math import pi, sqrt, sin, cos, fabs
from scipy import ndimage as ndi  # FAST connected components

# --- OSMnx SETTINGS -----------------------------------------------------------
ox.settings.use_cache   = True
ox.settings.log_console = False
ox.settings.timeout     = 180

# --- CONFIG (CITY / PCCA) -----------------------------------------------------
GIANT_MIN_FRAC        = 0.55   # local mode bias for "big-enough" cluster
MAX_LOCAL_THRESHOLDS  = 40     # max thresholds scanned in local window (smaller = faster)
IRREGULAR_BOUNDARY    = True

# Registries (re-use across runs)
if "graphs_by_radius" not in globals():
    graphs_by_radius = {}
if "boundaries_by_radius" not in globals():
    boundaries_by_radius = {}
if "nodes_by_radius" not in globals():
    nodes_by_radius = {}
if "edges_by_radius" not in globals():
    edges_by_radius = {}
if "water_by_radius" not in globals():
    water_by_radius = {}

# Cache for country-wide threshold: {POP_RASTER_path: (best_thr, debug)}
if "_PCCA_COUNTRY_CACHE" not in globals():
    _PCCA_COUNTRY_CACHE = {}

# --- GENERIC HELPERS ----------------------------------------------------------
def _project_graph_version_safe(G):
    try:
        return ox.project_graph(G)
    except AttributeError:
        return ox.projection.project_graph(G)

def _get_largest_component_version_safe(G, strongly=False):
    if strongly and G.is_directed():
        comp = max(nx.strongly_connected_components(G), key=len)
    else:
        comp = max(nx.weakly_connected_components(G.to_directed()), key=len)
    return G.subgraph(comp).copy()

def load_street_network(center_latlon, distance_m: int, network_type: str = "drive"):
    kwargs = dict(center_point=center_latlon, dist=distance_m, network_type=network_type,
                  simplify=True, retain_all=False, truncate_by_edge=True)
    try:
        return ox.graph_from_point(**kwargs)
    except AttributeError:
        return ox.graph.graph_from_point(**kwargs)

def _count_spheroid_area(lat_rad, dLat_rad, dLng_rad):
    """Cell area on a WGS84-like spheroid (km²), as in Cao et al. 1_CCA.py."""
    a = 6378.137
    b = 6356.752314245179
    e = sqrt(1.0 - (b / a) ** 2)
    tmp = 1.0 - (e ** 2) * (sin(lat_rad) ** 2)
    m = a * (1.0 - e ** 2) / (tmp * sqrt(tmp))
    n = a / sqrt(tmp)
    area = dLat_rad * dLng_rad * m * n * cos(lat_rad)
    return fabs(area)

# --- FAST CCA USING SCIPY -----------------------------------------------------
STRUCT_8 = np.ones((3, 3), dtype=bool)  # 8-neighbour connectivity

def _cca_stats(thr_mask, row_areas):
    """
    thr_mask: boolean array (True = above threshold inside window)
    row_areas: 1D array of per-row pixel areas (km²)
    Returns: (cluster_areas (1D), labels (2D))  -- labels can be None if no cluster.
    """
    if not np.any(thr_mask):
        return None, None

    # Connected components (fast C+Fortran implementation)
    labels, nlab = ndi.label(thr_mask, structure=STRUCT_8)
    if nlab == 0:
        return None, None

    # Per-pixel areas
    pixel_areas = (row_areas[:, None] * thr_mask.astype(float))
    # Sum areas per label index (0 = background)
    areas = np.bincount(labels.ravel(), weights=pixel_areas.ravel())
    cluster_areas = areas[1:]  # drop background

    if cluster_areas.size == 0:
        return None, None

    return cluster_areas, labels

# --- OPTION A: COUNTRY-LEVEL THRESHOLD ----------------------------------------
def _pcca_country_cache_path(pop_raster_path: Path) -> Path:
    """
    Save cache next to the city's data folder.
    Assumes pop_raster_path is something like: inputs/<city>/data/pop.tif
    """
    pop_raster_path = Path(pop_raster_path)
    return pop_raster_path.parent / "pcca_country_thr.json"


def load_or_compute_country_threshold(pop_raster_path, max_threshold=None):
    pop_raster_path = Path(pop_raster_path)
    cache_path = _pcca_country_cache_path(pop_raster_path)

    # 1) Load if exists
    if cache_path.exists():
        with open(cache_path, "r") as f:
            j = json.load(f)
        best_thr = float(j["best_thr"])
        print(f"[PCCA-country] loaded cached threshold {best_thr:.1f} from {cache_path}")
        return best_thr, j.get("debug", None)

    # 2) Else compute once
    best_thr, debug = _pcca_country_threshold_from_pop_raster(pop_raster_path, max_threshold=max_threshold)

    # 3) Save for future runs
    payload = {"best_thr": float(best_thr), "debug": debug}
    with open(cache_path, "w") as f:
        json.dump(payload, f)
    print(f"[PCCA-country] saved cached threshold {best_thr:.1f} to {cache_path}")

    return best_thr, debug

def _pcca_country_threshold_from_pop_raster(pop_raster_path, max_threshold=None):
    """
    Scan ALL valid population cells in POP_RASTER (no distance mask),
    compute per-threshold cluster areas with fast CCA, and pick a global
    percolation threshold (Option A).
    """
    # Use cache if available
    if pop_raster_path in _PCCA_COUNTRY_CACHE:
        best_thr, debug = _PCCA_COUNTRY_CACHE[pop_raster_path]
        print(f"[PCCA-country] using cached threshold ≈ {best_thr:.1f}")
        return best_thr, debug

    t0 = time.time()
    with rasterio.open(pop_raster_path) as src:
        arr = src.read(1).astype(float)
        arr[arr <= 0] = np.nan
        h, w = arr.shape
        transform = src.transform

        mask_all = np.isfinite(arr) & (arr > 0)
        if not np.any(mask_all):
            raise RuntimeError("No valid population values in POP_RASTER.")

        values = np.unique(np.floor(arr[mask_all]))
        vals   = values[values > 0]
        if max_threshold is not None:
            vals = vals[vals <= max_threshold]
        if len(vals) == 0:
            raise RuntimeError("No positive thresholds for country-level PCCA.")

        # row-wise cell areas (km²)
        pixelWidth  = transform.a
        pixelHeight = transform.e
        dLat_rad = pixelHeight * pi / 180.0
        dLng_rad = pixelWidth  * pi / 180.0
        row_areas = np.zeros(h, dtype=float)
        for i in range(h):
            lat_deg = transform.f + (i + 0.5) * pixelHeight
            lat_rad = lat_deg * pi / 180.0
            row_areas[i] = _count_spheroid_area(lat_rad, dLat_rad, dLng_rad)

        thresholds   = []
        largest_frac = []
        entropies    = []

        for t in vals:
            thr_mask = (arr > t) & mask_all
            cluster_areas, labels = _cca_stats(thr_mask, row_areas)
            if cluster_areas is None:
                continue

            sizes = np.asarray(cluster_areas, float)
            total_area = sizes.sum()
            lf = float(sizes.max() / total_area)
            p  = sizes / total_area
            entropy = float(-np.sum(p * np.log(p + 1e-15)))

            thresholds.append(float(t))
            largest_frac.append(lf)
            entropies.append(entropy)

        thresholds   = np.asarray(thresholds,   dtype=float)
        largest_frac = np.asarray(largest_frac, dtype=float)
        entropies    = np.asarray(entropies,    dtype=float)

        if thresholds.size == 0:
            raise RuntimeError("Country-level PCCA produced no valid thresholds.")

        # Prefer thresholds where largest_frac is in [0.3, 0.95]
        band = (largest_frac >= 0.30) & (largest_frac <= 0.95)
        if np.any(band):
            idx_best = np.where(band)[0][np.argmax(entropies[band])]
        else:
            idx_best = int(np.argmax(entropies))

        best_thr = float(thresholds[idx_best])
        t1 = time.time()
        print(f"[PCCA-country] scanned {len(thresholds)} thresholds "
              f"in {t1 - t0:,.1f}s — best_thr≈{best_thr:.1f}")

        debug = {
            "thresholds": thresholds.tolist(),
            "largest_frac": largest_frac.tolist(),
            "entropy": entropies.tolist(),
            "best_index": int(idx_best),
            "best_threshold": best_thr,
        }

        _PCCA_COUNTRY_CACHE[pop_raster_path] = (best_thr, debug)
        return best_thr, debug

# --- OPTION B + LOCAL STEP FOR A: LOCAL PCCA BOUNDARY ------------------------
def _pcca_local_boundary_from_pop_raster(pop_raster_path,
                                         center_latlon,
                                         max_r_m=20_000,
                                         best_thr_fixed=None,
                                         giant_min_frac=0.55):
    """
    Local PCCA around CENTER_LATLON.

    If best_thr_fixed is None:
      - Fully local scan (Option B) over all thresholds within the radial window.

    If best_thr_fixed is not None:
      - Local scan ONLY over thresholds in a band centred on best_thr_fixed
        (up to MAX_LOCAL_THRESHOLDS). This keeps Option A fast, but lets
        each city adjust away from the exact country threshold.
    """
    lat0, lon0 = float(center_latlon[0]), float(center_latlon[1])

    with rasterio.open(pop_raster_path) as src:
        t0 = time.time()

        arr = src.read(1).astype(float)
        arr[arr <= 0] = np.nan
        h, w = arr.shape
        transform = src.transform
        src_crs   = src.crs

        # pixel centres in raster CRS
        xs = np.arange(w) * transform.a + transform.c + transform.a / 2.0
        ys = np.arange(h) * transform.e + transform.f + transform.e / 2.0
        xv, yv = np.meshgrid(xs, ys)

        # distances to centre in metres (WebMercator)
        (mx,), (my,) = _reproj_pts("EPSG:4326", "EPSG:3857", [lon0], [lat0])
        x_m, y_m = _reproj_pts(src_crs, "EPSG:3857",
                               xv.ravel().tolist(), yv.ravel().tolist())
        x_m = np.asarray(x_m, float).reshape(arr.shape)
        y_m = np.asarray(y_m, float).reshape(arr.shape)
        dist_m = np.hypot(x_m - mx, y_m - my)

        mask = (dist_m <= max_r_m) & np.isfinite(arr) & (arr > 0)
        if not np.any(mask):
            raise RuntimeError("No valid population data near centre within max_r_m.")

        # row-wise cell areas (km²)
        pixelWidth  = transform.a
        pixelHeight = transform.e
        dLat_rad = pixelHeight * pi / 180.0
        dLng_rad = pixelWidth  * pi / 180.0
        row_areas = np.zeros(h, dtype=float)
        for i in range(h):
            lat_deg = transform.f + (i + 0.5) * pixelHeight
            lat_rad = lat_deg * pi / 180.0
            row_areas[i] = _count_spheroid_area(lat_rad, dLat_rad, dLng_rad)

        # ---------- choose threshold set for local scan ---------------------
        values = np.unique(np.floor(arr[mask]))
        vals_all = values[values > 0]
        if vals_all.size == 0:
            raise RuntimeError("No positive thresholds for local PCCA.")
        vals_all = np.sort(vals_all)

        if best_thr_fixed is None:
            # Option B: scan all local thresholds (can be slower)
            thr_candidates = vals_all
        else:
            # Option A: scan only thresholds near the country threshold
            i0 = int(np.argmin(np.abs(vals_all - best_thr_fixed)))
            half = MAX_LOCAL_THRESHOLDS // 2
            lo = max(0, i0 - half)
            hi = min(len(vals_all), i0 + half + 1)
            thr_candidates = vals_all[lo:hi]
            if thr_candidates.size == 0:
                thr_candidates = np.array([best_thr_fixed], dtype=float)

        thresholds   = []
        largest_frac = []
        entropies    = []

        # ---------- scan thresholds locally (fast CCA) ----------------------
        for t in thr_candidates:
            thr_mask = (arr > t) & mask
            cluster_areas, labels_dummy = _cca_stats(thr_mask, row_areas)
            if cluster_areas is None:
                continue

            sizes = np.asarray(cluster_areas, float)
            total_area = sizes.sum()
            lf = float(sizes.max() / total_area)
            p  = sizes / total_area
            H  = float(-np.sum(p * np.log(p + 1e-15)))

            thresholds.append(float(t))
            largest_frac.append(lf)
            entropies.append(H)

        thresholds   = np.asarray(thresholds,   dtype=float)
        largest_frac = np.asarray(largest_frac, dtype=float)
        entropies    = np.asarray(entropies,    dtype=float)

        if thresholds.size == 0:
            raise RuntimeError("Local PCCA produced no valid thresholds.")

        # knee of largest_frac curve
        if thresholds.size >= 2:
            dlf = np.diff(largest_frac) / np.maximum(np.diff(thresholds), 1e-6)
            idx_knee = int(np.argmax(dlf))
            idx_knee = max(0, min(idx_knee + 1, thresholds.size - 1))
        else:
            idx_knee = 0

        idx_best = idx_knee

        # prefer thresholds where giant component is "big enough"
        if giant_min_frac is not None:
            eligible = np.where(largest_frac >= giant_min_frac)[0]
            if eligible.size > 0:
                idx_giant = eligible[np.argmax(entropies[eligible])]
                if entropies[idx_giant] > entropies[idx_best]:
                    idx_best = idx_giant

        best_thr = float(thresholds[idx_best])

        debug = {
            "thresholds": thresholds.tolist(),
            "largest_frac": largest_frac.tolist(),
            "entropy": entropies.tolist(),
            "best_index": int(idx_best),
            "best_threshold": best_thr,
        }

        # ---------- FINAL CCA AT best_thr → CITY CLUSTER --------------------
        thr_mask_best = (arr > best_thr) & mask
        cluster_areas, labels = _cca_stats(thr_mask_best, row_areas)
        if cluster_areas is None or labels is None:
            raise RuntimeError("No clusters at best local PCCA threshold.")

        # pick cluster containing the centre (in raster coords)
        dlon = xv - lon0
        dlat = yv - lat0
        dist_deg = np.hypot(dlon, dlat)
        ci, cj = np.unravel_index(dist_deg.argmin(), dist_deg.shape)
        center_cluster = labels[ci, cj]

        if center_cluster == 0:
            # fallback: largest cluster by area
            areas_pix = np.bincount(labels.ravel(), weights=thr_mask_best.ravel().astype(float))
            if areas_pix.size <= 1:
                raise RuntimeError("No non-empty clusters at best_thr in local PCCA.")
            center_cluster = int(np.argmax(areas_pix[1:]) + 1)

        cluster_mask = (labels == center_cluster)

        # polygonize → polygon in raster CRS
        shapes = list(
            features.shapes(cluster_mask.astype("uint8"),
                            mask=cluster_mask,
                            transform=transform)
        )
        from shapely.geometry import shape as _shape
        polys = [_shape(geom) for (geom, val) in shapes if int(val) == 1]
        if not polys:
            raise RuntimeError("Failed to derive polygon from local PCCA cluster mask.")
        poly = unary_union(polys)

        # project to 3857
        gseries   = gpd.GeoSeries([poly], crs=src_crs)
        gdf_3857  = gseries.to_frame(name="geometry").to_crs(3857)
        poly_3857 = gdf_3857.geometry.unary_union

        # effective radius from centre
        poly_for_R = poly_3857
        if isinstance(poly_for_R, MultiPolygon):
            poly_for_R = max(list(poly_for_R.geoms), key=lambda g: g.area)

        xs_poly, ys_poly = poly_for_R.exterior.coords.xy
        xs_poly = np.asarray(xs_poly, float)
        ys_poly = np.asarray(ys_poly, float)
        dR = np.hypot(xs_poly - mx, ys_poly - my)
        R_m = float(dR.max())

        t1 = time.time()
        print(f"[PCCA-local/{'A' if best_thr_fixed is not None else 'B'}] "
              f"scanned {len(thresholds)} thresholds in {t1 - t0:,.1f}s — "
              f"best_thr≈{best_thr:.1f}, R≈{R_m/1000:.2f} km")

        return poly_3857, best_thr, R_m, debug

# --- MAIN: CHOOSE MODE, BUILD BOUNDARY ---------------------------------------
assert 'POP_RASTER' in globals(), "POP_RASTER path not defined (set in PATHS cell)."

pcca_debug_country = None

if PCCA_MODE == "country":
    # Option A: global/country-level PCCA → then local refinement around city
    best_thr_country, pcca_debug_country = load_or_compute_country_threshold(
        POP_RASTER,
        max_threshold=None
    )
    pcca_poly_3857, pcca_thr, R_pcca_m, pcca_debug_local = _pcca_local_boundary_from_pop_raster(
        POP_RASTER,
        CENTER_LATLON,
        max_r_m=MAX_RADIUS_M,
        best_thr_fixed=best_thr_country,
        giant_min_frac=GIANT_MIN_FRAC
    )
    pcca_debug = {"country": pcca_debug_country, "local": pcca_debug_local}
elif PCCA_MODE == "local":
    # Option B: fully local PCCA (slower)
    pcca_poly_3857, pcca_thr, R_pcca_m, pcca_debug = _pcca_local_boundary_from_pop_raster(
        POP_RASTER,
        CENTER_LATLON,
        max_r_m=MAX_RADIUS_M,
        best_thr_fixed=None,
        giant_min_frac=GIANT_MIN_FRAC
    )
else:
    raise ValueError("PCCA_MODE must be 'country' or 'local'.")

R_GLOBAL_M = int(float(np.clip(R_pcca_m, 1000.0, MAX_RADIUS_M)))
print(f"[POP PCCA / {PCCA_MODE}] {PLACE_NAME}")
print(f"  best pop threshold ≈ {pcca_thr:.1f} people/km² | "
      f"derived R_GLOBAL_M = {R_GLOBAL_M/1000:.2f} km")

R      = R_GLOBAL_M
R_SHOW = R_GLOBAL_M

# --- DEBUG PLOTS (optional; keep if useful) -----------------------------------
if PCCA_MODE == "country" and pcca_debug_country is not None:
    thrs_c  = np.asarray(pcca_debug_country["thresholds"],   dtype=float)
    lfrac_c = np.asarray(pcca_debug_country["largest_frac"], dtype=float)
    ent_c   = np.asarray(pcca_debug_country["entropy"],      dtype=float)

    if thrs_c.size > 0:
        fig_dbg, ax_dbg = plt.subplots(1, 2, figsize=(10, 4))
        ax_dbg[0].plot(thrs_c, lfrac_c, marker='o', markersize=3, linewidth=1.0)
        ax_dbg[0].axvline(pcca_thr, color='k', linestyle='--', linewidth=1.0)
        ax_dbg[0].set_xlabel("Threshold (people/km²)")
        ax_dbg[0].set_ylabel("Largest cluster / total area")
        ax_dbg[0].set_title("Country percolation curve (population)")

        ax_dbg[1].plot(thrs_c, ent_c, marker='o', markersize=3, linewidth=1.0)
        ax_dbg[1].axvline(pcca_thr, color='k', linestyle='--', linewidth=1.0)
        ax_dbg[1].set_xlabel("Threshold (people/km²)")
        ax_dbg[1].set_ylabel("Shannon entropy")
        ax_dbg[1].set_title("Country cluster-size entropy")

        fig_dbg.suptitle(f"Country PCCA diagnostics — {PLACE_NAME}", fontsize=11)
        fig_dbg.tight_layout()
       # plt.show()

dbg_loc = pcca_debug["local"] if PCCA_MODE == "country" else pcca_debug
if dbg_loc.get("thresholds", []):
    thrs_l  = np.asarray(dbg_loc["thresholds"],   dtype=float)
    lfrac_l = np.asarray(dbg_loc["largest_frac"], dtype=float)
    ent_l   = np.asarray(dbg_loc["entropy"],      dtype=float)

    fig_dbg2, ax_dbg2 = plt.subplots(1, 2, figsize=(10, 4))
    ax_dbg2[0].plot(thrs_l, lfrac_l, marker='o', markersize=3, linewidth=1.0)
    ax_dbg2[0].axvline(pcca_thr, color='k', linestyle='--', linewidth=1.0)
    ax_dbg2[0].set_xlabel("Threshold (people/km²)")
    ax_dbg2[0].set_ylabel("Largest cluster / total area")
    ax_dbg2[0].set_title("Local percolation curve (population)")

    ax_dbg2[1].plot(ent_l, marker='o', markersize=3, linewidth=1.0)
    ax_dbg2[1].axvline(np.argmax(ent_l), color='k', linestyle='--', linewidth=1.0)
    ax_dbg2[1].set_xlabel("Index")
    ax_dbg2[1].set_ylabel("Shannon entropy")
    ax_dbg2[1].set_title("Local entropy (index)")

    fig_dbg2.suptitle(f"Local PCCA diagnostics ({PCCA_MODE} mode) — {PLACE_NAME}", fontsize=11)
    fig_dbg2.tight_layout()
   # plt.show()

# --- Streets and raw boundary (raw_bound = POP PCCA polygon) ------------------
t_net0 = time.time()
G_raw = load_street_network(CENTER_LATLON, distance_m=R, network_type="drive")
G_raw = _get_largest_component_version_safe(G_raw, strongly=False)
G     = _project_graph_version_safe(G_raw)
nodes, edges = ox.graph_to_gdfs(G)
if nodes.crs is None:
    nodes.set_crs(epsg=4326, inplace=True); edges.set_crs(epsg=4326, inplace=True)
if str(nodes.crs) != "EPSG:3857":
    nodes = nodes.to_crs(3857); edges = edges.to_crs(3857)
t_net1 = time.time()
print(f"[street network] fetched + projected in {t_net1 - t_net0:,.1f}s")

center_wgs  = gpd.GeoSeries([Point(CENTER_LATLON[1], CENTER_LATLON[0])], crs="EPSG:4326")
center_3857 = center_wgs.to_crs(3857).iloc[0]

raw_bound    = pcca_poly_3857
circle_bound = center_3857.buffer(R)

# --- COASTLINE → POLYGONIZE → LAND MASK --------------------------------------
t_coast0 = time.time()
expand = R + 4000
try:
    coast = ox.geometries_from_point(CENTER_LATLON, tags={"natural": "coastline"}, dist=expand)
except Exception:
    coast = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

if coast.crs is None:
    coast = coast.set_crs(epsg=4326)
coast = coast.to_crs(3857)
coast = coast[coast.geometry.type.isin(["LineString","MultiLineString"])].copy()
if not coast.empty:
    try:
        coast = coast.explode(index_parts=False).reset_index(drop=True)
    except TypeError:
        coast = coast.explode().reset_index(drop=True)

land_mask = None
if not coast.empty:
    minx, miny, maxx, maxy = raw_bound.bounds
    frame_margin = 3000.0
    frame = box(minx - frame_margin, miny - frame_margin,
                maxx + frame_margin, maxy + frame_margin)

    coast_lines = unary_union(list(coast.geometry.values))
    merged = linemerge(unary_union([coast_lines, frame.exterior]))
    merged = snap(merged, frame.boundary, 100)

    polys = list(polygonize(merged))
    polys_gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:3857") if polys else \
                gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    if not polys_gdf.empty:
        center_pt  = center_3857
        edges_3857 = edges
        keep_geoms = []

        try:
            sindex = edges_3857.sindex
        except Exception:
            sindex = None

        for geom in polys_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            is_land = False
            if geom.contains(center_pt):
                is_land = True
            else:
                if sindex is not None:
                    cand_idx = list(sindex.intersection(geom.bounds))
                    if cand_idx:
                        sub = edges_3857.iloc[cand_idx]
                        if sub.geometry.intersects(geom).any():
                            is_land = True
                else:
                    if edges_3857.geometry.intersects(geom).any():
                        is_land = True
            if is_land:
                keep_geoms.append(geom)

        if keep_geoms:
            land_mask = unary_union(keep_geoms)

if land_mask is not None and not land_mask.is_empty:
    lm = land_mask.buffer(0)
    final_bound = raw_bound.intersection(lm)
    if final_bound.is_empty:
        final_bound = raw_bound
else:
    final_bound = raw_bound

final_series = gpd.GeoSeries([final_bound], crs="EPSG:3857")
gdf_bound    = final_series.to_frame(name="geometry")
t_coast1 = time.time()
print(f"[coast trim] completed in {t_coast1 - t_coast0:,.1f}s")

# --- Save registries ---------------------------------------------------------
graphs_by_radius[R]     = G
boundaries_by_radius[R] = gdf_bound
nodes_by_radius[R]      = nodes
edges_by_radius[R]      = edges
water_by_radius[R]      = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
water_gdf               = water_by_radius[R]

print(f"[{R/1000:.2f} km] nodes={len(nodes):,} edges={len(edges):,} | "
      f"boundary area={gdf_bound.area.iloc[0]:,.0f} m²")

# --- Plot check (fast, wide view) --------------------------------------------
t_plot0 = time.time()
fig, ax = plt.subplots(figsize=(8, 8))

cx, cy = center_3857.x, center_3857.y
ax.set_xlim(cx - R, cx + R)
ax.set_ylim(cy - R, cy + R)

try:
    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
except Exception:
    pass

edges.plot(ax=ax, linewidth=0.45, color="#555555", alpha=0.9, zorder=5)

plot_raw   = gpd.GeoSeries([raw_bound], crs="EPSG:3857").simplify(30, preserve_topology=True)
plot_final = gdf_bound.geometry.simplify(30, preserve_topology=True)

gpd.GeoDataFrame(geometry=[circle_bound], crs="EPSG:3857").boundary.plot(
    ax=ax, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=4, label="R_GLOBAL circle"
)
plot_raw.boundary.plot(
    ax=ax, color="#ff8800", linewidth=1.0, linestyle="-.", zorder=6,
    label=f"raw POP PCCA ({PCCA_MODE})"
)
plot_final.boundary.plot(
    ax=ax, color="black", linewidth=1.4, zorder=7, label="trimmed (land)"
)

ax.legend(loc="lower right", frameon=True)
ax.set_title(f"POP-PCCA boundary [{PCCA_MODE}] — {PLACE_NAME} — R≈{R/1000:.2f} km")
ax.axis("off"); plt.tight_layout()
t_plot1 = time.time()
print(f"[plot] time = {t_plot1 - t_plot0:,.1f}s")
#plt.show()

# ===================== Notebook cell 8 =====================
# ─── PERFORMANCE PACK v2: CENTRALITY @ R_GLOBAL_M (Global + Local @ 7 km) ─────
# Network accessibility uses global and local centrality measures;
# sampled approximation is available for speed.
# Turn approximations off if you want exact values:
#   APPROX_GLOBAL = False  → exact NetworkX closeness
#   APPROX_LOCAL  = False  → exact local harmonic (all nodes as sources)

import time
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from scipy.sparse import csr_matrix
from networkx.algorithms.centrality import closeness_centrality

# Optional parallel (used only for exact local harmonic)
try:
    from joblib import Parallel, delayed
except Exception:
    import sys, subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "joblib"])
    from joblib import Parallel, delayed

# ====== TUNING ================================================================
SIGMA_K        = 1.0        # z-threshold for hubs
INTERSECT_TOL  = 20         # m, consolidate_intersections tolerance (↑ to shrink graph)
BATCH_SIZE     = 2048       # batch for full local harmonic (exact mode)
N_JOBS         = -1         # -1 = all cores
DTYPE          = np.float32 # friendly on speed/memory
R_LOCAL_FIXED  = 7000       # fixed local radius = 7 km

# ---- NEW: speed/accuracy knobs ----------------------------------------------
APPROX_GLOBAL        = True   # True = fast sampled "closeness", False = exact NetworkX
N_GLOBAL_SOURCES     = 400    # sample size for global approx
GLOBAL_RAND_SEED     = 123

APPROX_LOCAL         = False   # True = fast sampled local harmonic, False = exact
N_LOCAL_SOURCES      = 400    # sample size for local approx
LOCAL_RAND_SEED      = 42
# ============================================================================

if "R_GLOBAL_M" not in globals():
    raise RuntimeError("R_GLOBAL_M is not defined. Run the density-based boundary cell first.")
if "graphs_by_radius" not in globals() or R_GLOBAL_M not in graphs_by_radius:
    raise RuntimeError("Global graph not built for R_GLOBAL_M. Run the build-networks cell first.")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _largest_component(Gd, strongly=False):
    if strongly and Gd.is_directed():
        comp = max(nx.strongly_connected_components(Gd), key=len)
    else:
        comp = max(nx.weakly_connected_components(Gd.to_directed()), key=len)
    return Gd.subgraph(comp).copy()

def _to_sparse_3857_undirected(Gc_u, nodelist):
    try:
        A = nx.to_scipy_sparse_array(
            Gc_u, nodelist=nodelist, weight="length",
            dtype=DTYPE, format="csr"
        )
    except AttributeError:
        A = nx.to_scipy_sparse_matrix(
            Gc_u, nodelist=nodelist, weight="length",
            dtype=DTYPE, format="csr"
        )
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A, dtype=DTYPE)
    A.sort_indices()
    return A

def _consolidate_version_robust(G_in, tol_m):
    """Return (Gc, cnodes, cedges, orig_to_con)."""
    res = ox.consolidate_intersections(
        G_in, tolerance=tol_m, rebuild_graph=True, dead_ends=False
    )
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], (nx.MultiDiGraph, nx.DiGraph)):
        Gc, cmapping = res
        cnodes, cedges = ox.graph_to_gdfs(Gc)
    elif isinstance(res, tuple) and len(res) == 3:
        cnodes, cedges, cmapping = res
        Gc = ox.utils_graph.graph_from_gdfs(cnodes, cedges, graph_attrs=G_in.graph)
    elif isinstance(res, (nx.MultiDiGraph, nx.DiGraph)):
        Gc = res
        cnodes, cedges = ox.graph_to_gdfs(Gc)
        cmapping = Gc.graph.get("consolidation", {}).get("mapping", {})
    else:
        raise RuntimeError("Unexpected consolidate_intersections return type")

    # Build orig->con, fallback to nearest snapping if needed
    orig_to_con = {}
    if isinstance(cmapping, dict) and len(cmapping) > 0:
        for cnode, origs in cmapping.items():
            try:
                for o in origs:
                    orig_to_con[o] = cnode
            except TypeError:
                if isinstance(origs, dict) and "orig" in origs:
                    for o in origs["orig"]:
                        orig_to_con[o] = cnode

    if len(orig_to_con) < len(G_in):
        from osmnx import distance as oxdist
        onodes_gdf, _ = ox.graph_to_gdfs(G_in)
        xs = onodes_gdf.geometry.x.values
        ys = onodes_gdf.geometry.y.values
        con_ids = oxdist.nearest_nodes(Gc, xs, ys, return_dist=False)
        for o, c in zip(onodes_gdf.index.values, np.asarray(con_ids).ravel()):
            orig_to_con[o] = c

    return Gc, cnodes, cedges, orig_to_con

# ---- Exact local harmonic calculation ---------------------------------------
def _harmonic_from_sources(indices, A, limit):
    D = sp_dijkstra(A, directed=False, indices=indices, limit=limit,
                    return_predecessors=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        invD = 1.0 / D
        rows = np.arange(D.shape[0])
        invD[rows, indices] = 0.0          # remove self-distances
        invD[~np.isfinite(invD)] = 0.0
        cnt = (invD > 0).sum(axis=1)
        sm  = invD.sum(axis=1)
        out = np.divide(sm, cnt, out=np.zeros_like(sm), where=cnt > 0)
    return out

def _parallel_local_harmonic_all_nodes(A, limit, batch=BATCH_SIZE, n_jobs=N_JOBS):
    n = A.shape[0]
    blocks = [(i, min(i + batch, n)) for i in range(0, n, batch)]

    def job(lo, hi):
        return _harmonic_from_sources(np.arange(lo, hi), A, limit)

    parts = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(job)(lo, hi) for lo, hi in blocks
    )
    return np.concatenate(parts).astype(DTYPE, copy=False)

# ---- NEW: fast sampled local harmonic ---------------------------------------
def _approx_local_harmonic_sampled(A, limit, k_sources=400, seed=42):
    """
    Approximate local harmonic centrality at radius `limit`
    by sampling k_sources probe nodes.

    Returns: array of shape (N_nodes,) with harmonic scores.
    """
    n = A.shape[0]
    k = int(min(max(1, k_sources), n))
    rng = np.random.default_rng(seed)

    sources = rng.choice(n, size=k, replace=False)
    D = sp_dijkstra(A, directed=False, indices=sources, limit=limit,
                    return_predecessors=False)  # shape (k, n)

    with np.errstate(divide='ignore', invalid='ignore'):
        invD = 1.0 / D
        rows = np.arange(k)
        invD[rows, sources] = 0.0          # remove self-distances
        invD[~np.isfinite(invD)] = 0.0
        cnt = (invD > 0).sum(axis=0)       # per node
        sm  = invD.sum(axis=0)
        out = np.divide(sm, cnt, out=np.zeros_like(sm, dtype=DTYPE), where=cnt > 0)
    return out.astype(DTYPE, copy=False)

# ---- NEW: fast sampled "global closeness" -----------------------------------
def _approx_global_closeness_sampled(A, k_sources=400, seed=123):
    """
    Approximate global closeness: for each node j, we compute the
    average distance from j to k sampled "anchor" nodes. Closeness
    is then ~ 1 / mean distance.

    Returns: dict {node_index: score} in the *contracted* index space.
    """
    n = A.shape[0]
    k = int(min(max(1, k_sources), n))
    rng = np.random.default_rng(seed)

    sources = rng.choice(n, size=k, replace=False)
    D = sp_dijkstra(A, directed=False, indices=sources,
                    return_predecessors=False)   # (k, n)

    with np.errstate(divide='ignore', invalid='ignore'):
        D[D <= 0] = np.nan
        meanD = np.nanmean(D, axis=0)     # per node
        C = 1.0 / meanD
        C[~np.isfinite(C)] = 0.0

    # output as dict indexed by contracted-node index 0..n-1
    return {int(i): float(C[i]) for i in range(n)}

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL + LOCAL@7km on the density-derived global graph
# ──────────────────────────────────────────────────────────────────────────────
centrality_by_radius = {}
hubs_by_radius = {}
centrality_distributions = {}

R = int(R_GLOBAL_M)  # key used across your dicts

t0 = time.time()
G_cur   = graphs_by_radius[R]
nodes_g = nodes_by_radius[R]
bound   = boundaries_by_radius[R]
study_poly = bound.geometry.iloc[0]

# clip + largest component
keep_idx = nodes_g[nodes_g.geometry.within(study_poly)].index
G_clip = G_cur.subgraph(keep_idx).copy()
G_clip = _largest_component(G_clip, strongly=False)
t1 = time.time()
print(f"[centrality] clip+largest component: {t1 - t0:.2f} s | nodes kept={len(G_clip):,}")

# consolidate once → contracted graph
Gc, cnodes, cedges, orig_to_con = _consolidate_version_robust(G_clip, INTERSECT_TOL)
Gc_u = Gc.to_undirected(as_view=True)
nodelist_c = list(Gc_u.nodes())
idx_c = {n: i for i, n in enumerate(nodelist_c)}
t2 = time.time()
print(f"[centrality] consolidate: {t2 - t1:.2f} s | consolidated nodes={len(Gc):,}")

# CSR once
A = _to_sparse_3857_undirected(Gc_u, nodelist_c)
t3 = time.time()
print(f"[centrality] build sparse matrix: {t3 - t2:.2f} s | shape={A.shape}")

# ---- GLOBAL closeness (exact OR approximate) --------------------------------
if APPROX_GLOBAL:
    print(f"[centrality] Using APPROX global closeness with {N_GLOBAL_SOURCES} sampled sources.")
    Cg_dict_idx = _approx_global_closeness_sampled(
        A,
        k_sources=N_GLOBAL_SOURCES,
        seed=GLOBAL_RAND_SEED,
    )
    # convert from index-space to contracted node id space
    Cg_con = {n: Cg_dict_idx[idx_c[n]] for n in nodelist_c}
else:
    print("[centrality] Using EXACT global closeness (NetworkX).")
    Cg_con = closeness_centrality(Gc_u, distance="length", wf_improved=True)
t4 = time.time()
print(f"[centrality] global centrality step: {t4 - t3:.2f} s")

# ---- LOCAL harmonic at fixed radius (exact OR approximate) -------------------
if APPROX_LOCAL:
    print(f"[centrality] Using APPROX local harmonic with {N_LOCAL_SOURCES} sampled sources.")
    C_local_con_arr = _approx_local_harmonic_sampled(
        A,
        limit=R_LOCAL_FIXED,
        k_sources=N_LOCAL_SOURCES,
        seed=LOCAL_RAND_SEED,
    )
else:
    print("[centrality] Using EXACT local harmonic (all nodes as sources).")
    C_local_con_arr = _parallel_local_harmonic_all_nodes(
        A,
        limit=R_LOCAL_FIXED,
        batch=BATCH_SIZE,
        n_jobs=N_JOBS,
    )
t5 = time.time()
print(f"[centrality] local centrality step: {t5 - t4:.2f} s")

# map back to original nodes (inside boundary)
nodes_out = nodes_g.loc[keep_idx].copy()

nodes_out["C_global"] = nodes_out.index.map(
    lambda o: Cg_con.get(orig_to_con.get(o, None), 0.0)
).astype(float)

C_local_map = {cnode: C_local_con_arr[idx_c[cnode]] for cnode in nodelist_c}
nodes_out["C_local"] = nodes_out.index.map(
    lambda o: C_local_map.get(orig_to_con.get(o, None), 0.0)
).astype(float)

t6 = time.time()
print(f"[centrality] map back to original nodes: {t6 - t5:.2f} s")

# Hub classification from standardised centrality
mu_g = float(nodes_out["C_global"].mean()); sd_g = float(nodes_out["C_global"].std(ddof=0))
mu_l = float(nodes_out["C_local"].mean());  sd_l = float(nodes_out["C_local"].std(ddof=0))
zg = (nodes_out["C_global"] - mu_g) / (sd_g if sd_g > 0 else 1.0)
zl = (nodes_out["C_local"]  - mu_l) / (sd_l if sd_l > 0 else 1.0)
nodes_out["C_global_z"] = zg.astype(float)
nodes_out["C_local_z"]  = zl.astype(float)

thr_g = mu_g + SIGMA_K * sd_g
thr_l = mu_l + SIGMA_K * sd_l
hi_g  = nodes_out.loc[nodes_out["C_global_z"] >= SIGMA_K].copy()
hi_l  = nodes_out.loc[nodes_out["C_local_z"]  >= SIGMA_K].copy()

# registries
centrality_by_radius[R] = nodes_out
hubs_by_radius[R] = {
    "hi_g": hi_g,
    "hi_l": hi_l,
    "thr_g_abs": float(thr_g),
    "thr_l_abs": float(thr_l),
    "mu_g": mu_g, "sd_g": sd_g,
    "mu_l": mu_l, "sd_l": sd_l,
    "sigma_k": float(SIGMA_K),
    "approx_global": bool(APPROX_GLOBAL),
    "approx_local":  bool(APPROX_LOCAL),
    "N_global_sources": int(N_GLOBAL_SOURCES),
    "N_local_sources":  int(N_LOCAL_SOURCES),
}

centrality_distributions[R] = {
    "C_global": nodes_out["C_global"].to_numpy(),
    "C_local":  nodes_out["C_local"].to_numpy(),
    "summary": {
        "N_nodes": int(len(nodes_out)),
        "mu_g": mu_g, "sd_g": sd_g,
        "mu_l": mu_l, "sd_l": sd_l,
        "p_global_ge_+1sigma": float((nodes_out["C_global_z"] >= SIGMA_K).mean()),
        "p_local_ge_+1sigma":  float((nodes_out["C_local_z"]  >= SIGMA_K).mean()),
    }
}

# For downstream compatibility
R_LOCAL_OPT = R_LOCAL_FIXED
hubs_by_radius[R]["hi_l_opt"]    = hi_l
hubs_by_radius[R]["R_local_opt"] = int(R_LOCAL_OPT)

t7 = time.time()
print(
    f"[centrality@{R/1000:.1f} km] TOTAL={t7 - t0:.2f} s | "
    f"N={len(nodes_out):,} | contracted={len(Gc):,} | "
    f"μg={mu_g:.4g} σg={sd_g:.4g} thr_g={thr_g:.4g} | "
    f"μl={mu_l:.4g} σl={sd_l:.4g} thr_l={thr_l:.4g} | "
    f"hubs_g(z≥{SIGMA_K})={len(hi_g):,} | hubs_l(z≥{SIGMA_K})={len(hi_l):,} | "
    f"local R = {R_LOCAL_FIXED/1000:.1f} km | "
    f"approx_global={APPROX_GLOBAL} | approx_local={APPROX_LOCAL}"
)

# ===================== Notebook cell 9 =====================
# ─── PLOTTING: Distributions & Map (Global vs Local @ 7 km) ───
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from scipy.stats import gaussian_kde

# ========= sanity checks =========
assert "R_GLOBAL_M" in globals(), "R_GLOBAL_M is missing. Run the density-based boundary step first."
assert "centrality_distributions" in globals() and R_GLOBAL_M in centrality_distributions, \
    "Missing centrality_distributions[R_GLOBAL_M]. Run the centrality cell first."
assert "hubs_by_radius" in globals() and R_GLOBAL_M in hubs_by_radius, \
    "Missing hubs_by_radius[R_GLOBAL_M]. Run the centrality cell first."
assert "boundaries_by_radius" in globals() and R_GLOBAL_M in boundaries_by_radius, \
    "Missing boundaries_by_radius[R_GLOBAL_M]. Build networks/boundary first."
assert "edges_by_radius" in globals() and R_GLOBAL_M in edges_by_radius, \
    "Missing edges_by_radius[R_GLOBAL_M]. Build networks first."

# fixed 7 km local radius (to align with your computation cell)
R_LOCAL_FIXED = 7000
R_LOCAL_OPT = R_LOCAL_FIXED

# ========= helpers =========
def _kde_line(data):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) < 10 or np.std(data) == 0:
        return None, None
    kde = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 300)
    ys = kde(xs)
    return xs, ys

def _to_crs(target, gdf):
    if gdf is None or getattr(gdf, "empty", False):
        return gdf
    if not hasattr(gdf, "crs") or gdf.crs is None:
        return gdf
    return gdf if gdf.crs == target else gdf.to_crs(target)

def _get_water_for_R(R, target_crs):
    """Fetch water polygons for radius R from water_by_radius (if present) and CRS-harmonize."""
    if "water_by_radius" not in globals():
        return None
    w = water_by_radius.get(R, None)
    if w is None or getattr(w, "empty", False):
        return None
    return _to_crs(target_crs, w)

# ========= (A) DISTRIBUTIONS & MAP @ R_GLOBAL_M =========
R = int(R_GLOBAL_M)
nbins = 40

Cg = centrality_distributions[R]["C_global"]
Cl = centrality_distributions[R]["C_local"]
summ = centrality_distributions[R]["summary"]

# A1) DISTRIBUTIONS (Global vs Local@3 km)
fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.2))

ax.hist(Cg, bins=nbins, density=True, alpha=0.30, edgecolor="none", color="#d62728", label="Global")
ax.hist(Cl, bins=nbins, density=True, alpha=0.30, edgecolor="none", color="#1f77b4", label="Local (7 km)")

xs, ys = _kde_line(Cg)
if xs is not None: ax.plot(xs, ys, linewidth=1.5, color="#d62728")
xs, ys = _kde_line(Cl)
if xs is not None: ax.plot(xs, ys, linewidth=1.5, color="#1f77b4")

mu_g, sd_g = summ["mu_g"], summ["sd_g"]
ax.axvline(mu_g,        color="#d62728", linestyle="--", linewidth=1)
ax.axvline(mu_g+sd_g,   color="#d62728", linestyle=":",  linewidth=1)

ax.set_title(f"Centrality distributions @ {R/1000:.1f} km (Local = 7 km)", fontsize=10)
ax.tick_params(axis='both', labelsize=8)
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
#plt.show()

# A2) MAP: Global hubs (red) + Local hubs (orange)
fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2))
bound = boundaries_by_radius[R]
edges = edges_by_radius[R]
hubs  = hubs_by_radius[R]
hi_g  = hubs["hi_g"]
hi_l  = hubs["hi_l"]

target_crs = bound.crs
edges_p = _to_crs(target_crs, edges)
hi_g_p  = _to_crs(target_crs, hi_g)
hi_l_p  = _to_crs(target_crs, hi_l)
water_p = _get_water_for_R(R, target_crs)

minx, miny, maxx, maxy = bound.total_bounds
ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

try:
    ctx.add_basemap(ax, crs=target_crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
except Exception as e:
    print(f"[basemap {R}] skipped:", e)

bound.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=1)
if water_p is not None and not water_p.empty:
    water_p.plot(ax=ax, color="steelblue", alpha=0.35, zorder=3, label="Water")
if edges_p is not None and not edges_p.empty:
    edges_p.plot(ax=ax, linewidth=0.35, alpha=0.6, color="#777777", zorder=2)
if hi_l_p is not None and not hi_l_p.empty:
    hi_l_p.plot(ax=ax, markersize=6, alpha=0.90, color="#ff9900", zorder=10,
                label=f"Local hubs (z≥1σ) @ {R_LOCAL_FIXED/1000:.1f} km")
if hi_g_p is not None and not hi_g_p.empty:
    hi_g_p.plot(ax=ax, markersize=8, alpha=0.92, color="#e31a1c", zorder=11,
                label=f"Global hubs ({R/1000:.1f} km)")

ax.set_axis_off()
ax.set_title(f"Hubs (Global {R/1000:.1f} km vs Local 7 km)", fontsize=12, pad=8)

handles, labels = ax.get_legend_handles_labels()
labels_clean, handles_clean = [], []
for h, l in zip(handles, labels):
    if l and not l.startswith("_"):
        labels_clean.append(l); handles_clean.append(h)
if labels_clean:
    ax.legend(handles_clean, labels_clean, loc="upper left", fontsize=8, frameon=True)

plt.tight_layout()
#plt.show()

# ===================== Notebook cell 14 =====================
hub_points = hubs.copy()

# ===================== Notebook cell 17 =====================
import numpy as np
import pandas as pd
import geopandas as gpd
from hdbscan import HDBSCAN

# ─── Load hubs from centrality results ─────────────────────
assert "hubs_by_radius" in globals() and int(R_GLOBAL_M) in hubs_by_radius, \
    "Run the centrality step first (to populate hubs_by_radius[R_GLOBAL_M])."

R = int(R_GLOBAL_M)
hi_g = hubs_by_radius[R].get("hi_g", gpd.GeoDataFrame(geometry=[]))  # Global hubs
hi_l = hubs_by_radius[R].get("hi_l", gpd.GeoDataFrame(geometry=[]))  # Local hubs

# ─── Determine CRS ────────────────────────────────────────
if "boundaries_by_radius" in globals() and R in boundaries_by_radius:
    target_crs = boundaries_by_radius[R].crs
elif hasattr(hi_g, "crs") and hi_g.crs is not None:
    target_crs = hi_g.crs
else:
    target_crs = "EPSG:3857"

def _to_crs(target, gdf):
    if gdf is None or getattr(gdf, "empty", False):
        return gdf
    return gdf if gdf.crs == target else gdf.to_crs(target)

hi_g = _to_crs(target_crs, hi_g)
hi_l = _to_crs(target_crs, hi_l)

# ─── Optional: Clip to boundary ───────────────────────────
if "boundaries_by_radius" in globals() and R in boundaries_by_radius:
    def _clip(gdf):
        return gpd.clip(gdf, boundaries_by_radius[R]) if (gdf is not None and not gdf.empty) else gdf
    hi_g = _clip(hi_g)
    hi_l = _clip(hi_l)

frames = []

# ─── (1) GLOBAL HUBS: HDBSCAN clustering (no singletons per point) ─
if hi_g is not None and not hi_g.empty:
    g = hi_g.copy()
    for col in ("cluster_tag", "l_label", "g_label"):
        if col in g.columns:
            g = g.drop(columns=[col])

    g["scale"] = "global"

    XYg = np.column_stack((g.geometry.x.values, g.geometry.y.values))
    g_labels = HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        cluster_selection_method="eom"
    ).fit_predict(XYg)

    g["g_label"] = g_labels
    g["cluster_tag"] = [("G", int(x)) if x != -1 else None for x in g["g_label"]]

    # Assign one fallback tag for all global noise — use Series to avoid tuple/ndarray assignment error
    g_noise = g["cluster_tag"].isna()
    if g_noise.any():
        next_id_g = (max([x[1] for x in g["cluster_tag"] if x is not None], default=-1) + 1)
        idx = g.index[g_noise]
        g.loc[idx, "cluster_tag"] = pd.Series([("G", next_id_g)] * len(idx), index=idx, dtype=object)

    frames.append(g[["geometry", "scale", "cluster_tag"]])

# ─── (2) LOCAL HUBS: HDBSCAN clustering ───────
if hi_l is not None and not hi_l.empty:
    l = hi_l.copy()
    for col in ("cluster_tag", "l_label", "g_label"):
        if col in l.columns:
            l = l.drop(columns=[col])

    l["scale"] = "local"

    XYl = np.column_stack((l.geometry.x.values, l.geometry.y.values))
    l_labels = HDBSCAN(
        min_cluster_size=10,
        min_samples=4,
        cluster_selection_method="eom"
    ).fit_predict(XYl)

    l["l_label"] = l_labels
    l["cluster_tag"] = [("L", int(x)) if x != -1 else None for x in l["l_label"]]

    # Assign one fallback tag for all local noise — Series assignment avoids tuple broadcast issues
    l_noise = l["cluster_tag"].isna()
    if l_noise.any():
        next_id_l = (max([x[1] for x in l["cluster_tag"] if x is not None], default=-1) + 1)
        idx = l.index[l_noise]
        l.loc[idx, "cluster_tag"] = pd.Series([("L", next_id_l)] * len(idx), index=idx, dtype=object)

    frames.append(l[["geometry", "scale", "cluster_tag"]])

# ─── COMBINE ALL HUB POINTS ────────────────────────────────
hub_points = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=target_crs)

# ─── Coerce tags to strict tuples BEFORE ID mapping ────────
def _to_tuple_tag(val):
    if isinstance(val, tuple) and len(val) == 2 and val[0] in ("G", "L"):
        return (val[0], int(val[1]))
    if isinstance(val, str) and len(val) >= 2 and val[0] in ("G", "L") and val[1:].lstrip("-").isdigit():
        return (val[0], int(val[1:]))
    # anything else → treat as local noise to avoid spurious clusters
    return ("L", -1)

hub_points["cluster_tag"] = hub_points["cluster_tag"].apply(_to_tuple_tag)

# ─── Map tuple tags → contiguous int IDs (no string concat) ─
def _pair_key(t):
    return (t[0], int(t[1]))

unique_pairs = sorted(hub_points["cluster_tag"].unique(), key=_pair_key)
pair_to_id = {pair: i for i, pair in enumerate(unique_pairs)}
hub_points["cluster"] = hub_points["cluster_tag"].map(pair_to_id).astype(int)

print(f"[Seeded hubs @ {R/1000:.1f} km] total hub points: {len(hub_points)}")
print("  pre-merge clusters K =", hub_points["cluster"].nunique())

# ─── OPTIONAL: Gentle merge nearby clusters (global + local) ────────
DO_POST_MERGE = True
MERGE_CENTROID_DIST_M = 450.0
MERGE_HULL_GAP_M = 120.0

if DO_POST_MERGE and hub_points["cluster"].nunique() > 1:
    cluster_hulls = hub_points.dissolve(by="cluster").convex_hull
    hulls_gdf = gpd.GeoDataFrame({"cluster": cluster_hulls.index.astype(int)},
                                 geometry=cluster_hulls.values, crs=hub_points.crs)
    centroids = hulls_gdf.geometry.centroid
    n = len(hulls_gdf)

    adj = np.zeros((n, n), dtype=bool)
    coords = np.column_stack((centroids.x.values, centroids.y.values))

    for i in range(n):
        for j in range(i + 1, n):
            gi = hulls_gdf.geometry.iloc[i]
            gj = hulls_gdf.geometry.iloc[j]
            touch_or_intersect = gi.touches(gj) or gi.intersects(gj)
            d2 = (coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2
            close_centroid = d2 <= MERGE_CENTROID_DIST_M ** 2
            gap = gi.distance(gj)
            close_hulls = gap <= MERGE_HULL_GAP_M
            if touch_or_intersect or close_centroid or close_hulls:
                adj[i, j] = adj[j, i] = True

    # Union-Find to fuse clusters
    parent = {i: i for i in range(n)}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                union(i, j)

    idx_to_cluster = hulls_gdf["cluster"].tolist()
    cluster_to_group = {idx_to_cluster[i]: find(i) for i in range(n)}
    hub_points["cluster"] = hub_points["cluster"].map(cluster_to_group).astype(int)

    # Reindex to contiguous IDs
    uniq = sorted(hub_points["cluster"].unique())
    remap = {old: i for i, old in enumerate(uniq)}
    hub_points["cluster"] = hub_points["cluster"].map(remap).astype(int)

    print(f"[post-merge] clusters fused → K = {hub_points['cluster'].nunique()}")

# ===================== Notebook cell 18 =====================
# ─── VORONOI FROM (MERGED) HUB CLUSTERS — ADAPTIVE CORE HULLS (BIG vs SMALL HUBS) ───
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
from shapely.ops import voronoi_diagram

# --- inputs ---
assert 'hub_points' in globals(), "Missing hub_points (run the clustering cell first)."
assert 'R_GLOBAL_M' in globals(), "Missing R_GLOBAL_M (run the radius/boundary cell first)."
R = int(R_GLOBAL_M)

# Boundary/CRS
if 'boundaries_by_radius' in globals() and R in boundaries_by_radius:
    gdf_bound_3857_use = boundaries_by_radius[R]
else:
    raise RuntimeError(f"Global boundary not found for R_GLOBAL_M={R}.")

envelope   = gdf_bound_3857_use.geometry.iloc[0]
target_crs = gdf_bound_3857_use.crs

def _to_crs(target, gdf):
    if gdf is None or getattr(gdf, "empty", False):
        return gdf
    return gdf if (hasattr(gdf, "crs") and gdf.crs == target) else gdf.to_crs(target)

# Hub points CRS
hub_points = _to_crs(target_crs, hub_points)

# Street edges (context)
edges_ref = edges_by_radius.get(R, None) if 'edges_by_radius' in globals() else None
edges_ref = _to_crs(target_crs, edges_ref) if edges_ref is not None else None

# Water (optional)
water_plot = None
water_union = None
if 'water_by_radius' in globals() and R in water_by_radius:
    water_plot = _to_crs(target_crs, water_by_radius[R])
    if water_plot is not None and not water_plot.empty:
        try:
            water_union = water_plot.dissolve().geometry.iloc[0].buffer(0)
        except Exception:
            water_union = water_plot.unary_union.buffer(0)

# ---------------------------------------------------------------------
# 1) Cluster-wise hulls (big hubs use core hull, small hubs use full hull)
# ---------------------------------------------------------------------
if hub_points.empty or "cluster" not in hub_points.columns:
    raise RuntimeError("hub_points is empty or lacks 'cluster' labels.")

def _largest_part(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = [g for g in getattr(geom, "geoms", []) if not g.is_empty and g.geom_type == "Polygon"]
        if not parts:
            return geom
        return max(parts, key=lambda g: g.area)
    return geom

def _core_hull_for_cluster(pts_gdf, frac=0.7, min_core=30):
    """
    Hull from radial core of cluster:
    - centroid,
    - keep closest `frac` of points if ≥ min_core,
    - else use all points.
    """
    if pts_gdf.empty:
        return Polygon()

    xs = pts_gdf.geometry.x.values
    ys = pts_gdf.geometry.y.values
    coords = np.column_stack((xs, ys))
    cx, cy = coords.mean(axis=0)
    d = np.hypot(xs - cx, ys - cy)

    if len(pts_gdf) >= min_core:
        r_cut = np.quantile(d, frac)
        core = pts_gdf[d <= r_cut]
        if len(core) >= 3:
            hull = core.unary_union.convex_hull
        else:
            hull = pts_gdf.unary_union.convex_hull
    else:
        hull = pts_gdf.unary_union.convex_hull

    return _largest_part(hull.buffer(0))

clusters = sorted(hub_points["cluster"].unique())
hull_rows = []
cluster_sizes = {}  # number of hub nodes per cluster

BIG_CLUSTER_MIN_N = 400  # ≥ this many hubs → treat as big

for cid in clusters:
    pts_c = hub_points[hub_points["cluster"] == cid]
    n_c = len(pts_c)
    cluster_sizes[int(cid)] = n_c

    if n_c >= BIG_CLUSTER_MIN_N:
        # big hub: core hull (avoid giant stretched polygon)
        hull = _core_hull_for_cluster(pts_c, frac=0.7, min_core=30)
    else:
        # small/medium hubs: full hull keeps them visible
        hull = _largest_part(pts_c.unary_union.convex_hull.buffer(0))

    hull_rows.append({"cluster": int(cid), "geometry": hull})

hulls_gdf = gpd.GeoDataFrame(hull_rows, crs=target_crs).reset_index(drop=True)

# Seeds inside hull (one per cluster)
seed_points = hulls_gdf.geometry.representative_point()
seeds_gdf = gpd.GeoDataFrame({"cluster": hulls_gdf["cluster"].astype(int)},
                             geometry=seed_points.values, crs=target_crs)

# ---------------------------------------------------------------------
# 2) Voronoi from seeds, assign each cell to the seed it CONTAINS (fallback to nearest)
# ---------------------------------------------------------------------
if len(seeds_gdf) == 1:
    vor_gdf = gpd.GeoDataFrame({"cluster": [int(seeds_gdf.loc[0, "cluster"])]},
                               geometry=[envelope], crs=target_crs)
else:
    multi = MultiPoint(list(seeds_gdf.geometry))
    vor = voronoi_diagram(multi, envelope=envelope, edges=False)
    vor_polys = [poly.buffer(0) for poly in vor.geoms]
    vor_gdf = gpd.GeoDataFrame(geometry=vor_polys, crs=target_crs)

    cell_clusters = []
    for poly in vor_gdf.geometry:
        inside = seeds_gdf[seeds_gdf.geometry.within(poly)]
        if len(inside) == 1:
            cell_clusters.append(int(inside.iloc[0]["cluster"]))
        elif len(inside) > 1:
            c = poly.centroid
            j = seeds_gdf.distance(c).idxmin()
            cell_clusters.append(int(seeds_gdf.loc[j, "cluster"]))
        else:
            c = poly.centroid
            j = seeds_gdf.distance(c).idxmin()
            cell_clusters.append(int(seeds_gdf.loc[j, "cluster"]))
    vor_gdf["cluster"] = cell_clusters
    vor_gdf = gpd.clip(vor_gdf, gdf_bound_3857_use)

# ---------------------------------------------------------------------
# 3) Build inner/outer polygons — adaptive targets for big vs small hubs
# ---------------------------------------------------------------------
# Base parameters
INNER_MIN_AREA_M2       = 5_000.0   # absolute minimum inner area
# For big hubs:
INNER_MIN_RATIO_BIG     = 0.02      # fraction of Voronoi cell area
INNER_MAX_HULL_MULT_BIG = 1.5
# For small hubs:
INNER_MIN_RATIO_SMALL   = 0.06      # make small hubs visually larger
INNER_MAX_HULL_MULT_SMALL = 3.0

HULL_BUFFER_STEPS_M     = [20, 40, 80, 150, 250, 400]
SEED_RADII_MULTS        = [1.0, 1.5, 2.0]
EDGE_EPS_M              = 0.01
SMOOTH_BUFFER_M         = 40.0
FINAL_SHRINK_M          = 5.0

def _smooth_clean(geom, smooth_m=SMOOTH_BUFFER_M):
    if geom.is_empty:
        return geom
    if smooth_m <= 0:
        return geom
    return _largest_part(geom.buffer(smooth_m).buffer(-smooth_m))

def _grow_inner_to_target(cell_poly, hull_poly, seed_pt, n_pts):
    cell_poly = _largest_part(cell_poly.buffer(0))
    cell_area = max(cell_poly.area, 1.0)

    # choose regime based on cluster size
    if n_pts >= BIG_CLUSTER_MIN_N:
        ratio   = INNER_MIN_RATIO_BIG
        max_mul = INNER_MAX_HULL_MULT_BIG
    else:
        ratio   = INNER_MIN_RATIO_SMALL
        max_mul = INNER_MAX_HULL_MULT_SMALL

    hull_in_cell = _largest_part(cell_poly.intersection(hull_poly).buffer(0))
    hull_area = hull_in_cell.area if not hull_in_cell.is_empty else hull_poly.area

    target_area = max(INNER_MIN_AREA_M2, ratio * cell_area)
    if hull_area > 0:
        target_area = min(target_area, max_mul * hull_area)

    inner = hull_in_cell
    if not inner.is_empty and inner.area >= 0.6 * target_area:
        inner = _smooth_clean(inner)
        if inner.area >= 0.8 * target_area:
            return inner

    for d in HULL_BUFFER_STEPS_M:
        cand = _largest_part(cell_poly.intersection(hull_poly.buffer(d)).buffer(0))
        cand = _smooth_clean(cand)
        if not cand.is_empty and cand.area >= target_area:
            return cand

    base_r = max(40.0, np.sqrt(target_area / np.pi))
    for m in SEED_RADII_MULTS:
        r = base_r * m
        cand = _largest_part(cell_poly.intersection(seed_pt.buffer(r)).buffer(0))
        cand = _smooth_clean(cand)
        if not cand.is_empty and cand.area >= target_area * 0.9:
            return cand

    shrink = cell_poly.buffer(-FINAL_SHRINK_M)
    if not shrink.is_empty and shrink.area >= target_area * 0.5:
        return _largest_part(_smooth_clean(shrink))
    return inner if not inner.is_empty else _largest_part(cell_poly.buffer(0))

inner_polys, outer_polys, cell_ids = [], [], []
for _, cell in vor_gdf.iterrows():
    cid = int(cell["cluster"])
    cell_poly = cell.geometry
    hull_poly = hulls_gdf.loc[hulls_gdf["cluster"] == cid, "geometry"].iloc[0]
    seed_pt   = seeds_gdf.loc[seeds_gdf["cluster"] == cid, "geometry"].iloc[0]
    n_pts     = cluster_sizes.get(cid, 0)

    inner = _grow_inner_to_target(cell_poly, hull_poly, seed_pt, n_pts)
    if not inner.is_empty and FINAL_SHRINK_M > 0:
        inner = _largest_part(inner.buffer(-FINAL_SHRINK_M))

    outer = _largest_part(cell_poly.difference(inner).buffer(0))

    inner_polys.append(inner)
    outer_polys.append(outer)
    cell_ids.append(cid)

inner_gdf = gpd.GeoDataFrame({"cluster": cell_ids}, geometry=inner_polys, crs=target_crs)
outer_gdf = gpd.GeoDataFrame({"cluster": cell_ids}, geometry=outer_polys, crs=target_crs)

# ---------------------------------------------------------------------
# 4) Optional water cut; if inner shrinks below target, re-grow inside water-cut cell
# ---------------------------------------------------------------------
if water_union is not None and not getattr(water_union, "is_empty", False):
    try:
        vor_gdf["geometry"]   = vor_gdf.geometry.difference(water_union).buffer(0)
        inner_gdf["geometry"] = inner_gdf.geometry.difference(water_union).buffer(0)
        outer_gdf["geometry"] = outer_gdf.geometry.difference(water_union).buffer(0)

        repaired = []
        for cid in inner_gdf["cluster"].astype(int):
            cell_poly = vor_gdf.loc[vor_gdf["cluster"] == cid, "geometry"].unary_union
            seed_pt   = seeds_gdf.loc[seeds_gdf["cluster"] == cid, "geometry"].iloc[0]
            hull_poly = hulls_gdf.loc[hulls_gdf["cluster"] == cid, "geometry"].iloc[0]
            geom      = inner_gdf.loc[inner_gdf["cluster"] == cid, "geometry"].iloc[0]
            n_pts     = cluster_sizes.get(cid, 0)

            # recompute target threshold
            if n_pts >= BIG_CLUSTER_MIN_N:
                ratio = INNER_MIN_RATIO_BIG
            else:
                ratio = INNER_MIN_RATIO_SMALL
            thresh = max(INNER_MIN_AREA_M2, ratio * max(cell_poly.area, 1.0)) * 0.9

            if geom.is_empty or geom.area < thresh:
                new_inner = _grow_inner_to_target(cell_poly, hull_poly, seed_pt, n_pts)
                repaired.append(new_inner)
            else:
                repaired.append(_largest_part(geom))
        inner_gdf["geometry"] = repaired

        vor_gdf   = vor_gdf[~vor_gdf.is_empty].copy()
        outer_gdf = outer_gdf[~outer_gdf.is_empty].copy()
    except Exception as e:
        print("[water cut] subtraction failed; polygons kept uncut:", e)

# ---------------------------------------------------------------------
# 5) Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 14))

gdf_bound_3857_use.boundary.plot(ax=ax, color="black", linewidth=1.4, zorder=1,
                                 label=f"Global boundary ({R/1000:.1f} km)")
if edges_ref is not None and not edges_ref.empty:
    edges_ref.plot(ax=ax, linewidth=0.35, alpha=0.6, color="#888888", zorder=2, label="_edges")
if water_plot is not None and not water_plot.empty:
    water_plot.plot(ax=ax, color="steelblue", alpha=0.35, zorder=3, label="Water")

if not outer_gdf.empty:
    outer_gdf.boundary.plot(ax=ax, color="#cc3333", linewidth=1.1, zorder=5, label="Voronoi (outer)")

if not inner_gdf.empty:
    inner_gdf.plot(ax=ax, facecolor="orange", alpha=0.55, edgecolor="#222222",
                   linewidth=1.2, zorder=6, label="Hub (inner)")

if not hub_points.empty:
    hub_points.plot(ax=ax, color="#e31a1c", markersize=5, alpha=0.9, zorder=7, label="Hub nodes")

minx, miny, maxx, maxy = gdf_bound_3857_use.total_bounds
ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
ax.set_axis_off()
ax.set_title("Voronoi from Cluster Seeds — Adaptive Inner Polygons", fontsize=16, pad=10)
try:
    ctx.add_basemap(ax, crs=target_crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
except Exception as e:
    print("[basemap] skipped:", e)

handles, labels = ax.get_legend_handles_labels()
labels_clean, handles_clean = [], []
for h, l in zip(handles, labels):
    if l and not l.startswith("_"):
        labels_clean.append(l); handles_clean.append(h)
if labels_clean:
    ax.legend(handles_clean, labels_clean, loc="upper left", fontsize=9, frameon=True)

plt.tight_layout()
#plt.show()

# ── outputs for downstream:
# hub_points (GeoDataFrame with 'cluster' ints), vor_gdf, inner_gdf, outer_gdf

# ===================== Notebook cell 19 =====================
# ─── A. USE HUBS AS "ACTIVITY FEATURES" (robust to hub_points being a dict) — R_GLOBAL_M aware ───
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# Resolve the active global radius key once
if 'R_GLOBAL_M' not in globals():
    raise RuntimeError("R_GLOBAL_M missing. Run the density-based radius cell first.")
R = int(R_GLOBAL_M)

def _extract_hub_gdf():
    """
    Return a GeoDataFrame with columns: geometry, cluster
    Handles cases where `hub_points` is:
      - already a GeoDataFrame
      - a dict containing a GeoDataFrame under common keys
      - a pandas DataFrame with a shapely geometry column
    If HDBSCAN wasn't run, builds a temporary 2-cluster proxy from hi_g/hi_l at R.
    """
    # 1) If hub_points exists
    if "hub_points" in globals():
        hp = hub_points
        # a) already a GeoDataFrame with 'cluster'
        if isinstance(hp, gpd.GeoDataFrame) and "cluster" in hp.columns:
            return hp
        # b) dict-like: try common keys
        if isinstance(hp, dict):
            for k in ("gdf", "geodataframe", "points", "data", "df"):
                if k in hp and isinstance(hp[k], gpd.GeoDataFrame) and "cluster" in hp[k].columns:
                    return hp[k]
        # c) pandas DataFrame with shapely geometries
        if isinstance(hp, pd.DataFrame) and "geometry" in hp.columns and "cluster" in hp.columns:
            try:
                return gpd.GeoDataFrame(hp, geometry="geometry", crs=getattr(hp, "crs", None))
            except Exception:
                pass

    # 2) Fallback: look for common variable names
    for name in ("hub_points_gdf", "hub_points_merged", "hub_nodes"):
        if name in globals():
            obj = globals()[name]
            if isinstance(obj, gpd.GeoDataFrame) and "cluster" in obj.columns:
                return obj

    # 3) Final fallback: temporary 2-cluster proxy from hi_g/hi_l (if HDBSCAN not run yet)
    if "hubs_by_radius" in globals() and R in hubs_by_radius:
        hi_g = hubs_by_radius[R].get("hi_g", gpd.GeoDataFrame(geometry=[]))
        hi_l = hubs_by_radius[R].get("hi_l", gpd.GeoDataFrame(geometry=[]))
        if isinstance(hi_g, gpd.GeoDataFrame) and not hi_g.empty:
            tmp = []
            g = hi_g.copy(); g["cluster"] = 0; tmp.append(g[["geometry","cluster"]])
            if isinstance(hi_l, gpd.GeoDataFrame) and not hi_l.empty:
                l = hi_l.copy(); l["cluster"] = 1; tmp.append(l[["geometry","cluster"]])
            gdf = gpd.GeoDataFrame(pd.concat(tmp, ignore_index=True), crs=hi_g.crs)
            print(f"[warn] No clustered `hub_points` found. Built a temporary 2-cluster proxy from hi_g/hi_l @ R={R/1000:.1f} km.")
            print("       For proper results, run the HDBSCAN cell and re-run this.")
            return gdf

    # Otherwise: give a clear error
    raise RuntimeError("`hub_points` not found as a clustered GeoDataFrame. "
                       "Please run the HDBSCAN cell so `hub_points` exists with a 'cluster' column.")

# Get the hubs GeoDataFrame
hp_gdf = _extract_hub_gdf().copy()

# ── NEW: optional water-aware land clip at the active global radius ──
if ("boundaries_by_radius" in globals() and R in boundaries_by_radius):
    bound = boundaries_by_radius[R]
    # harmonize CRS
    if hp_gdf.crs is None and bound.crs is not None:
        hp_gdf.set_crs(bound.crs, inplace=True)
    if hp_gdf.crs != bound.crs:
        hp_gdf = hp_gdf.to_crs(bound.crs)

    if "water_by_radius" in globals() and R in water_by_radius:
        water = water_by_radius[R]
        if water is not None and not getattr(water, "empty", False):
            if water.crs != bound.crs:
                water = water.to_crs(bound.crs)
            # Build land = boundary – water (robust to topology)
            try:
                land_geom = bound.geometry.unary_union.difference(water.unary_union.buffer(0))
            except Exception:
                land_geom = bound.unary_union
                try:
                    land_geom = land_geom.difference(water.unary_union.buffer(0))
                except Exception:
                    pass
            # Keep only hubs on land
            hp_gdf = hp_gdf[hp_gdf.within(gpd.GeoSeries([land_geom], crs=bound.crs).iloc[0])].copy()

# Copy hubs as activity_feats (points), keep geometry and cluster labels
activity_feats = hp_gdf.copy()

# Ensure integer labels
activity_feats["cluster"] = activity_feats["cluster"].astype(int)

# Shift to 1..K (to match older downstream expectations)
activity_feats["cluster"] = activity_feats["cluster"] + 1

# ─── B. RECOMPUTE CENTROIDS PER CLUSTER ────────────────────────────
valid = activity_feats.copy()
centroids_series = valid.dissolve(by="cluster").centroid
centroid_gdf_new = gpd.GeoDataFrame(
    {"cluster": centroids_series.index.astype(int)},
    geometry=centroids_series.values,
    crs=activity_feats.crs
).reset_index(drop=True)

# ─── C. INTER-HUB DISTANCE ANALYSIS (cluster centroids) ────────────
if len(centroid_gdf_new) >= 2:
    centroid_coords = np.array([(pt.x, pt.y) for pt in centroid_gdf_new.geometry])
    dist_matrix = pdist(centroid_coords)
    avg_interhub_dist = float(np.mean(dist_matrix))
    max_interhub_dist = float(np.max(dist_matrix))
else:
    avg_interhub_dist = np.nan
    max_interhub_dist = np.nan

print(" INTER-HUB DISTANCE ANALYSIS (from hub cluster centroids)")
print(f"   Average distance between hubs: {avg_interhub_dist/1000:.2f} km" if np.isfinite(avg_interhub_dist) else "   (insufficient hubs)")
print(f"   Maximum distance between hubs: {max_interhub_dist/1000:.2f} km" if np.isfinite(max_interhub_dist) else "")

# ─── D. HUB NODE COUNTS PER CLUSTER ───────────────────────────────
poi_counts = (
    valid["cluster"]
    .value_counts()
    .sort_index()
)

print("\n HUB NODE COUNTS PER CLUSTER")
for cluster_id, count in poi_counts.items():
    print(f"  Cluster {int(cluster_id)}: {int(count)} hub-nodes")

# ─── E. VISUALIZE COUNTS AS BAR CHART ──────────────────────────────
plt.figure(figsize=(6, 4))
poi_counts.plot(kind="bar", color="skyblue")
plt.title("Number of Hub Nodes per Cluster")
plt.xlabel("Cluster ID")
plt.ylabel("Hub-node count")
plt.grid(True, axis="y", alpha=0.4)
plt.tight_layout()
#plt.show()

# ===================== Notebook cell 20 =====================
# R0a — BOOTSTRAP CONFIG (auto-uses city from your earlier setup)

import os, re

# Prefer explicitly set CITY_NAME; else fall back to PLACE_NAME or 'place'
assert ('PLACE_NAME' in globals()) or ('place' in globals()) or ('CITY_NAME' in globals()), \
    "Please define PLACE_NAME (or place) earlier."

CITY_NAME = globals().get('CITY_NAME',
             globals().get('PLACE_NAME',
               globals().get('place', 'Unknown City')))
CITY_NAME = str(CITY_NAME).strip()

def _slugify(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

CITY_SLUG = _slugify(CITY_NAME)

SHARED_OUT = "shared_data"
CITY_OUT   = "outputs"
os.makedirs(SHARED_OUT, exist_ok=True)
os.makedirs(CITY_OUT,   exist_ok=True)
CITY_OUT = str(OUT_DIR)
SHARED_OUT = str(OUT_DIR)  # or PROJECT_ROOT/"shared_data"/CITY_SLUG/run_xxx if you prefer

TICKS_PER_DAY = globals().get('TICKS_PER_DAY', 24)

print(f"CITY_NAME = {CITY_NAME} | CITY_SLUG = {CITY_SLUG}")
print(f"SHARED_OUT = {SHARED_OUT} | CITY_OUT = {CITY_OUT}")

# ===================== Notebook cell 21 =====================
# Zipf/entropy/CV diagnostics and per-city outputs
# - Requires R0a bootstrap run first (CITY_NAME, CITY_SLUG, SHARED_OUT).
# - Uses density-derived global radius (R_GLOBAL_M) if available; falls back to 7 km.

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import linregress
from scipy.spatial.distance import pdist

assert 'CITY_NAME' in globals() and 'CITY_SLUG' in globals() and 'SHARED_OUT' in globals(), \
    "Run the R0a bootstrap cell first."
assert 'activity_feats' in globals() and not activity_feats.empty, \
    "activity_feats not found (build it from hub_points in the previous cell)."

# ─────────────────────────────────────────────────────────────────────────────
# Radius-aware helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_crs(target, gdf):
    if gdf is None or getattr(gdf, "empty", False): 
        return gdf
    return gdf if (hasattr(gdf, "crs") and gdf.crs == target) else gdf.to_crs(target)

def _pick_radius_key():
    """Prefer R_GLOBAL_M if present in dicts; else 7_000; else any key."""
    r_candidates = []
    if 'R_GLOBAL_M' in globals():
        r_candidates.append(int(R_GLOBAL_M))
    r_candidates.append(7_000)
    if 'boundaries_by_radius' in globals():
        have = list(boundaries_by_radius.keys())
        for r in r_candidates:
            if r in have:
                return r
        if have:
            return sorted(have)[0]
    return None

R_KEY = _pick_radius_key()

# pick boundary (trimmed) and water layers
boundary_gdf = None
if 'boundaries_by_radius' in globals() and R_KEY in boundaries_by_radius:
    boundary_gdf = boundaries_by_radius[R_KEY]
elif 'gdf_bound_3857' in globals():
    boundary_gdf = gdf_bound_3857  # compatible field name

water_any = None
if 'water_by_radius' in globals() and R_KEY in water_by_radius:
    water_any = water_by_radius[R_KEY]
elif 'water_gdf' in globals():
    water_any = water_gdf  # compatible source

# ─────────────────────────────────────────────────────────────────────────────
# Optional land clip (exclude hubs on water, keep inside boundary if present)
# ─────────────────────────────────────────────────────────────────────────────
af = activity_feats.copy()
if isinstance(boundary_gdf, gpd.GeoDataFrame) and not boundary_gdf.empty:
    target_crs = boundary_gdf.crs
    if af.crs is None and target_crs is not None:
        af.set_crs(target_crs, inplace=True)
    if af.crs != target_crs:
        af = af.to_crs(target_crs)

    if isinstance(water_any, gpd.GeoDataFrame) and not water_any.empty:
        w = _to_crs(target_crs, water_any)
        try:
            land = gpd.overlay(boundary_gdf[['geometry']], w[['geometry']], how='difference')
        except Exception:
            # robust fallback: dissolve before difference
            land = gpd.GeoDataFrame(geometry=[boundary_gdf.unary_union.difference(w.unary_union)], crs=target_crs)
        # keep only hubs on land
        try:
            af = gpd.sjoin(af, land[['geometry']], predicate='within', how='inner').drop(columns=['index_right'])
        except Exception:
            af = af[af.within(land.unary_union)]
    else:
        # at least clip to boundary
        try:
            af = gpd.sjoin(af, boundary_gdf[['geometry']], predicate='within', how='inner').drop(columns=['index_right'])
        except Exception:
            af = af[af.within(boundary_gdf.unary_union)]

# If everything got filtered out (rare), revert so diagnostics still run
if isinstance(af, gpd.GeoDataFrame) and af.empty:
    af = activity_feats.copy()

# ─── proceed with diagnostics on `af` ────────────────────────────────────────

# Counts per hub-cluster (proxy for hub 'mass')
counts = (
    af["cluster"]
    .value_counts()
    .sort_values(ascending=False)
)
k = int(len(counts))
total_nodes = int(counts.sum())

# Rank-size (Zipf), entropy, and inter-hub dispersion
if k >= 2 and total_nodes > 0:
    ranks = np.arange(1, k + 1)
    slope, intercept, r_value, p_value, std_err = linregress(
        np.log(ranks), np.log(counts.values)
    )

    # Entropy (evenness of hub sizes)
    proportions = counts / total_nodes
    entropy = float(-np.sum(proportions * np.log(proportions)))
    max_entropy = float(np.log(k))
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else np.nan

    # Inter-hub distance dispersion (centroid spacing)
    centroids_series = af.dissolve(by="cluster").centroid
    centroid_gdf_new = gpd.GeoDataFrame(
        {"cluster": centroids_series.index.astype(int)},
        geometry=centroids_series.values,
        crs=af.crs
    )
    hub_coords = np.array([(pt.x, pt.y) for pt in centroid_gdf_new.geometry])
    if len(hub_coords) >= 2:
        dists = pdist(hub_coords)
        mean_dist = float(np.mean(dists))
        std_dist = float(np.std(dists))
        interhub_cv = float(std_dist / mean_dist) if mean_dist > 0 else np.nan
    else:
        mean_dist, interhub_cv = np.nan, np.nan
else:
    slope = np.nan; entropy = np.nan; max_entropy = np.nan; normalized_entropy = np.nan
    mean_dist = np.nan; interhub_cv = np.nan

# Simple dominance metrics
share_top1 = float(counts.iloc[0] / total_nodes) if total_nodes > 0 else np.nan
share_top2 = float(counts.iloc[:2].sum() / total_nodes) if total_nodes > 0 else np.nan

# Heuristic classification (unchanged)
if np.isfinite(share_top1) and np.isfinite(normalized_entropy):
    if (share_top1 >= 0.45 and normalized_entropy <= 0.70) or k == 1:
        structure_label = "Monocentric"
    elif (share_top1 < 0.30 and normalized_entropy >= 0.85 and k >= 3):
        structure_label = "Polycentric"
    else:
        structure_label = "Transitional / Weakly Polycentric"
else:
    structure_label = "Undetermined"

# ─── PRINT DIAGNOSTIC INDICATORS ───────────────────────────────────
print(" MONOCENTRICITY VS POLYCENTRICITY (HUB-BASED)\n")
print(f"1) Rank-Size Slope (Zipf): {slope if np.isfinite(slope) else float('nan'):.2f}")
print("   Steep (<< -1): Monocentric | Flat (~ -0.5): Polycentric")
print(f"\n2) Entropy: {entropy if np.isfinite(entropy) else float('nan'):.3f} "
      f"(Normalized: {normalized_entropy if np.isfinite(normalized_entropy) else float('nan'):.3f})")
print("   Low (<0.70): Monocentric | High (≥0.85): Polycentric")
print(f"\n3) Inter-Hub Distance CV: {interhub_cv if np.isfinite(interhub_cv) else float('nan'):.3f} "
      f"(Mean: {(mean_dist/1000.0) if np.isfinite(mean_dist) else float('nan'):.2f} km)")
print(f"\n4) Dominance: top1 share={share_top1:.2f} | top2 share={share_top2:.2f} | hubs K={k}")
print(f"\n→ STRUCTURE (heuristic): {structure_label}")

# ─── SAVE ZIPF/ENTROPY/CV FOR THIS CITY (column names match aux notebook) ────
zipf_out = pd.DataFrame([{
    "city": CITY_NAME,
    "city_slug": CITY_SLUG,
    "clusters_k": k,
    "total_hub_nodes": total_nodes,
    "zipf_slope": float(slope) if np.isfinite(slope) else np.nan,
    "entropy": float(entropy) if np.isfinite(entropy) else np.nan,
    "normalized_entropy": float(normalized_entropy) if np.isfinite(normalized_entropy) else np.nan,
    "interhub_cv": float(interhub_cv) if np.isfinite(interhub_cv) else np.nan,
    "mean_interhub_km": float(mean_dist/1000.0) if np.isfinite(mean_dist) else np.nan,
    "share_top1": share_top1,
    "share_top2": share_top2,
    "structure_label": structure_label,
}])
out_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_zipf.csv")
zipf_out.to_csv(OUT_DIR /out_path, index=False)
print(f"[ZIPF/HUB] saved → {out_path}")

# ===================== Notebook cell 22 =====================
# Global Moran’s I for HUB Cluster Labels (point-based) — radius-aware, water-aware, NumPy 2–safe
import numpy as np
import geopandas as gpd  # <-- needed

# Install deps if missing
try:
    import libpysal
    from esda.moran import Moran
except Exception:
    import sys, subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "libpysal", "esda"])
    import libpysal
    from esda.moran import Moran

assert 'activity_feats' in globals() and not activity_feats.empty, \
    "activity_feats must exist (created from hub_points)."

def _to_crs(target, gdf):
    if gdf is None or getattr(gdf, "empty", False):
        return gdf
    return gdf if (hasattr(gdf, "crs") and gdf.crs == target) else gdf.to_crs(target)

def _pick_radius_key():
    """Prefer R_GLOBAL_M; else 7_000; else any available key in boundaries_by_radius."""
    cand = []
    if 'R_GLOBAL_M' in globals(): cand.append(int(R_GLOBAL_M))
    cand.append(7_000)
    if 'boundaries_by_radius' in globals():
        have = list(boundaries_by_radius.keys())
        for r in cand:
            if r in have:
                return r
        if have:
            return sorted(have)[0]
    return None

R_KEY = _pick_radius_key()

pts = activity_feats.copy()

# Boundary & water (radius-aware)
boundary = None
if 'boundaries_by_radius' in globals() and R_KEY in boundaries_by_radius:
    boundary = boundaries_by_radius[R_KEY]
elif 'gdf_bound_3857' in globals():
    boundary = gdf_bound_3857

water_any = None
if 'water_by_radius' in globals() and R_KEY in water_by_radius:
    water_any = water_by_radius[R_KEY]
elif 'water_gdf' in globals():
    water_any = water_gdf

# Apply land mask (boundary – water) or clip to boundary if available
if isinstance(boundary, gpd.GeoDataFrame) and not boundary.empty:
    target_crs = boundary.crs
    pts = _to_crs(target_crs, pts)
    if isinstance(water_any, gpd.GeoDataFrame) and not water_any.empty:
        w = _to_crs(target_crs, water_any)
        try:
            land = gpd.overlay(boundary[['geometry']], w[['geometry']], how='difference')
        except Exception:
            b_union = boundary.union_all() if hasattr(boundary, "union_all") else boundary.unary_union
            w_union = w.union_all() if hasattr(w, "union_all") else w.unary_union
            land = gpd.GeoDataFrame(geometry=[b_union.difference(w_union)], crs=target_crs)
        try:
            pts = gpd.sjoin(pts, land[['geometry']], predicate='within', how='inner').drop(columns=['index_right'])
        except Exception:
            pts = pts[pts.within(land.union_all() if hasattr(land, "union_all") else land.unary_union)]
    else:
        try:
            pts = gpd.sjoin(pts, boundary[['geometry']], predicate='within', how='inner').drop(columns=['index_right'])
        except Exception:
            pts = pts[pts.within(boundary.union_all() if hasattr(boundary, "union_all") else boundary.unary_union)]

# If filtering removed everything, fall back so the stat can still run
if pts.empty:
    pts = activity_feats.copy()

# --- Build coordinates and KNN weights ---------------------------------------
coords = np.array([(pt.x, pt.y) for pt in pts.geometry])

print("Global Moran’s I for Hub Clusters")

# Need ≥ 2 points
if len(coords) < 2:
    print("Not enough points after land filtering (need ≥ 2).")
else:
    # Degenerate (all coincide) → tiny jitter to avoid singular KNN
    if np.allclose(np.ptp(coords, axis=0), 0.0):
        coords = coords + np.random.normal(scale=1e-3, size=coords.shape)

    # KNN weights (row-standardized)
    k_neighbors = min(5, max(1, len(coords) - 1))
    w = libpysal.weights.KNN.from_array(coords, k=k_neighbors)
    w.transform = "R"

    # Variable = numeric cluster label (1..K)
    y = pts["cluster"].values.astype(float)

    # Run Moran with permutations (default 999)
    moran = Moran(y, w)

    # Prefer permutation-based stats when available
    has_perm = hasattr(moran, "p_sim") and hasattr(moran, "z_sim")
    p_val = float(moran.p_sim) if has_perm else float(moran.p_norm)
    z_val = float(moran.z_sim) if has_perm else float(moran.z_norm)

    print(f"I value: {moran.I:.4f}")
    print(f"Expected I (randomization): {moran.EI:.4f}")
    print(f"p-value: {p_val:.4f} ({'significant' if p_val < 0.05 else 'not significant'})")
    print(f"z-score: {z_val:.2f}")

# ===================== Notebook cell 23 =====================
# ─── SYNTHETIC FACILITIES (NO FIXED TOTALS) — RADIUS-AWARE + WATER-AWARE + PUBLIC TOILETS ───
# Places Drinking Water, Schools, Markets, Public Toilets inside each inner polygon.
# Counts per hub depend ONLY on (i) number of street nodes and (ii) summed centrality
# of nodes within that hub — median-scaled, no global totals.
#
# Outputs kept for ABM compatibility:
#   dwater_gdf, schools_gdf, markets_gdf, toilets_gdf
#   drinking_points  (water point collection)

import random
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import osmnx as ox
import networkx as nx
from shapely.geometry import Point

# ========= TUNE (weights only; NO totals) =====================================
ALPHA_WATER    = 0.60   # 1.0 → only node count, 0.0 → only centrality
ALPHA_SCHOOL   = 0.55
ALPHA_MARKET   = 0.45
ALPHA_TOILET   = 0.50

# Which node centrality to aggregate within each hub:
CENTRICITY_MODE = "max"   # 'max' | 'mean' | 'global' | 'local'

# Minimum per-hub counts (ensure presence) and optional caps (None = no cap)
MIN_WATER_PER_HUB   = 1;  MAX_WATER_PER_HUB   = None
MIN_SCHOOL_PER_HUB  = 0;  MAX_SCHOOL_PER_HUB  = None
MIN_MARKET_PER_HUB  = 1;  MAX_MARKET_PER_HUB  = None
MIN_TOILET_PER_HUB  = 0;  MAX_TOILET_PER_HUB  = None

# Random placement
MAX_ATTEMPTS_PER_POINT = 40
RANDOM_SEED = 42
# ============================================================================

rng = random.Random(RANDOM_SEED)
assert 'inner_gdf' in globals() and not inner_gdf.empty, "Missing inner_gdf (run the Voronoi step first)."

# ---- pick the working radius (prefer R_GLOBAL_M, else 7km, else any available) ----
def _pick_radius_key():
    cand = []
    if 'R_GLOBAL_M' in globals(): cand.append(int(R_GLOBAL_M))
    cand.append(7_000)
    if 'boundaries_by_radius' in globals():
        have = list(boundaries_by_radius.keys())
        for r in cand:
            if r in have:
                return r
        if have:
            return sorted(have)[0]
    # last resort: if we only have standalone nodes/edges variables
    return None

R_KEY = _pick_radius_key()

# Prefer nodes/edges for the chosen radius; else fall back to generic
if (R_KEY is not None) and ('nodes_by_radius' in globals()) and (R_KEY in nodes_by_radius):
    nodes_cent = nodes_by_radius[R_KEY].copy()
else:
    assert "nodes" in globals(), "No nodes GeoDataFrame found."
    nodes_cent = nodes.copy()

# ─── Ensure centrality columns exist; if missing, compute them now (robust) ───
missing_cols = [c for c in ("C_global", "C_local") if c not in nodes_cent.columns]
if missing_cols:
    # Prefer the already-computed centrality_by_radius table to avoid an expensive
    # second betweenness calculation in the facility-allocation cell.
    reused = False
    if (R_KEY is not None) and ('centrality_by_radius' in globals()) and (R_KEY in centrality_by_radius):
        cent_src = centrality_by_radius[R_KEY]
        if cent_src is not None and not getattr(cent_src, "empty", True):
            for col in ("C_global", "C_local"):
                if col in cent_src.columns:
                    nodes_cent[col] = cent_src[col].reindex(nodes_cent.index).fillna(0.0).astype(float).values
            reused = all(c in nodes_cent.columns for c in ("C_global", "C_local"))
            if reused:
                print(f"[centrality] reused centrality_by_radius[{R_KEY}] for facility allocation.")

    if not reused:
        # Try to get a graph for the same radius, then fallback to generic G
        if (R_KEY is not None) and ('graphs_by_radius' in globals()) and (R_KEY in graphs_by_radius):
            G_for_c = graphs_by_radius[R_KEY]
        elif "G" in globals():
            G_for_c = G
        else:
            raise AssertionError(
                "'C_global'/'C_local' missing and no graph found to compute them. "
                "Make sure graphs_by_radius[R_KEY] or G exists."
            )
        G_und = G_for_c.to_undirected(as_view=False)

        # Light but robust: betweenness ~ C_global, degree ~ C_local.
        # NOTE: Betweenness is expensive, so this fallback is only reached if the
        # upstream centrality table is unavailable. Results are cached to disk.
        bc = betweenness_centrality_fast(G_und, weight='length', normalized=True, seed=0)
        degc = nx.degree_centrality(G_und)

        betw_s = pd.Series(bc, name="C_global")
        degc_s = pd.Series(degc, name="C_local")

        # align by index (handle osmid lists)
        if not nodes_cent.index.isin(betw_s.index).any() and "osmid" in nodes_cent.columns:
            osmid_series = nodes_cent["osmid"].apply(lambda x: x[0] if isinstance(x, (list, tuple, set)) and len(x) else x)
            nodes_cent = nodes_cent.set_index(osmid_series)
            nodes_cent.index.name = "node_id"

        nodes_cent["C_global"] = betw_s.reindex(nodes_cent.index).fillna(0.0).astype(float).values
        nodes_cent["C_local"]  = degc_s.reindex(nodes_cent.index).fillna(0.0).astype(float).values

        print(f"[centrality] computed on-the-fly for {len(nodes_cent):,} nodes "
              f"(betweenness→C_global, degree→C_local).")

# Harmonize CRS
target_crs = inner_gdf.crs
if nodes_cent.crs != target_crs:
    nodes_cent = nodes_cent.to_crs(target_crs)

# ─── Land mask (chosen boundary minus water) & water-safe inner polygons ───
def _to_crs(target, gdf):
    if gdf is None or getattr(gdf, "empty", False): return gdf
    return gdf if (hasattr(gdf, "crs") and gdf.crs == target) else gdf.to_crs(target)

# boundary & water for chosen radius (fallbacks supported)
bound_use = None
if (R_KEY is not None) and ('boundaries_by_radius' in globals()) and (R_KEY in boundaries_by_radius):
    bound_use = boundaries_by_radius[R_KEY]
elif 'gdf_bound_3857' in globals():
    bound_use = gdf_bound_3857

water_any = None
if (R_KEY is not None) and ('water_by_radius' in globals()) and (R_KEY in water_by_radius):
    water_any = water_by_radius[R_KEY]
elif 'water_gdf' in globals():
    water_any = water_gdf

inner_work = inner_gdf.copy()
land_mask = None
if isinstance(bound_use, gpd.GeoDataFrame) and not bound_use.empty:
    bound_use = _to_crs(target_crs, bound_use)
    if isinstance(water_any, gpd.GeoDataFrame) and not water_any.empty:
        water_any = _to_crs(target_crs, water_any)
        # Land = boundary – water (prefer overlay; fallback to union diff)
        try:
            land_mask = gpd.overlay(bound_use[['geometry']], water_any[['geometry']], how='difference')
        except Exception:
            b_union = bound_use.union_all() if hasattr(bound_use, "union_all") else bound_use.unary_union
            w_union = water_any.union_all() if hasattr(water_any, "union_all") else water_any.unary_union
            land_mask = gpd.GeoDataFrame(geometry=[b_union.difference(w_union)], crs=target_crs)
        # Clip inner polygons to land so facilities never fall on water
        try:
            inner_work = gpd.overlay(inner_gdf[['cluster','geometry']], land_mask[['geometry']], how='intersection')
        except Exception:
            inner_work = gpd.clip(inner_gdf, land_mask)
        # Filter nodes to land for fair allocation
        try:
            nodes_cent = gpd.sjoin(nodes_cent, land_mask[['geometry']], how='inner', predicate='within').drop(columns=['index_right'])
        except Exception:
            nodes_cent = nodes_cent[nodes_cent.within(land_mask.union_all() if hasattr(land_mask,"union_all") else land_mask.unary_union)]

# If clipping removed everything, fall back to original inner_gdf (rare)
if inner_work is None or inner_work.empty:
    inner_work = inner_gdf.copy()

# -------- helpers -------------------------------------------------------------
def _centrality_series(df: gpd.GeoDataFrame, mode: str):
    g = df["C_global"].astype(float)
    l = df["C_local"].astype(float)
    if mode == "global": return g
    if mode == "local":  return l
    if mode == "mean":   return 0.5 * (g + l)
    return np.maximum(g.values, l.values)  # default: 'max'

def _median_safe(x: np.ndarray):
    x = x[np.isfinite(x)]
    if x.size == 0: return 0.0
    return float(np.median(x))

def _ceil_clip(val, lo, hi=None):
    v = int(np.ceil(val))
    if v < lo: v = lo
    if (hi is not None) and v > hi: v = hi
    return v

def _random_points_in_polygon(poly, n, max_tries=MAX_ATTEMPTS_PER_POINT):
    if n <= 0: return []
    minx, miny, maxx, maxy = poly.bounds
    out, tries = [], 0
    while len(out) < n and tries < n * max_tries:
        pt = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(pt):
            out.append(pt)
        tries += 1
    return out

# -------- compute hub stats (node counts + centrality) ------------------------
rows = []
cent_field_sum = f"cent_sum_{CENTRICITY_MODE}"
for _, r in inner_work.iterrows():
    cid = int(r["cluster"])
    poly = r.geometry
    if poly.is_empty:
        continue
    nodes_here = nodes_cent.loc[nodes_cent.within(poly)]
    n_nodes = int(len(nodes_here))
    if n_nodes > 0:
        cvals = _centrality_series(nodes_here, CENTRICITY_MODE)
        c_sum = float(np.nansum(cvals))
        c_mean = float(np.nanmean(cvals))
    else:
        c_sum = 0.0
        c_mean = 0.0
    rows.append({"cluster": cid, "geometry": poly, "n_nodes": n_nodes, cent_field_sum: c_sum, "cent_mean": c_mean})

stats = gpd.GeoDataFrame(rows, crs=target_crs)
if stats.empty:
    raise RuntimeError("No inner (land) polygons or no nodes inside them.")

# Purely ratio-driven (median scaling)
med_nodes = _median_safe(stats["n_nodes"].to_numpy().astype(float))
med_cent  = _median_safe(stats[cent_field_sum].to_numpy().astype(float))

def _rel(x, m): return (x / m) if m > 0 else 0.0

def _counts_for(alpha, min_per, max_per):
    rel_nodes = stats["n_nodes"].apply(lambda v: _rel(float(v), med_nodes))
    rel_cent  = stats[cent_field_sum].apply(lambda v: _rel(float(v), med_cent))
    score = alpha * rel_nodes + (1.0 - alpha) * rel_cent
    return score.apply(lambda s: _ceil_clip(s, lo=min_per, hi=max_per if max_per is not None else None)).astype(int)

stats["alloc_water"]   = _counts_for(ALPHA_WATER,  MIN_WATER_PER_HUB,  MAX_WATER_PER_HUB)
stats["alloc_schools"] = _counts_for(ALPHA_SCHOOL, MIN_SCHOOL_PER_HUB, MAX_SCHOOL_PER_HUB)
stats["alloc_markets"] = _counts_for(ALPHA_MARKET, MIN_MARKET_PER_HUB, MAX_MARKET_PER_HUB)
stats["alloc_toilets"] = _counts_for(ALPHA_TOILET, MIN_TOILET_PER_HUB, MAX_TOILET_PER_HUB)

print("[alloc ratio] medians:",
      f"nodes_med={med_nodes:.1f}", f"cent_sum_med={med_cent:.3g}",
      "| totals → water:",   int(stats["alloc_water"].sum()),
      "schools:",            int(stats["alloc_schools"].sum()),
      "markets:",            int(stats["alloc_markets"].sum()),
      "toilets:",            int(stats["alloc_toilets"].sum()))

# -------- generate random points inside each (land) hub polygon ---------------
def _make_points(stats_df, col_name, feature_type):
    out_rows = []
    for _, rr in stats_df.iterrows():
        n = int(rr[col_name]); 
        if n <= 0: continue
        pts = _random_points_in_polygon(rr["geometry"], n)
        for p in pts:
            out_rows.append({"geometry": p, "feature_type": feature_type, "cluster": int(rr["cluster"])})
    return gpd.GeoDataFrame(out_rows, crs=target_crs)

dwater_gdf  = _make_points(stats, "alloc_water",   "Drinking Water")
schools_gdf = _make_points(stats, "alloc_schools", "School")
markets_gdf = _make_points(stats, "alloc_markets", "Market")
toilets_gdf = _make_points(stats, "alloc_toilets", "Public Toilet")

# Compatibility with the drinking-point representation
drinking_points = list(dwater_gdf.geometry) if not dwater_gdf.empty else []
BASELINE_TAP_WKBS = [pt.wkb for pt in drinking_points if pt is not None]
BASELINE_TAP_COUNT = len(BASELINE_TAP_WKBS)

# -------- quick plot (basemap Mapnik) ---------------
# choose boundary/edges for plotting at the same radius
if (R_KEY is not None) and ('boundaries_by_radius' in globals()) and (R_KEY in boundaries_by_radius):
    bound_plot = boundaries_by_radius[R_KEY]
else:
    bound_plot = gdf_bound_3857 if "gdf_bound_3857" in globals() else None

edges_plot = None
if (R_KEY is not None) and ('edges_by_radius' in globals()) and (R_KEY in edges_by_radius):
    edges_plot = edges_by_radius[R_KEY]
if edges_plot is not None and edges_plot.crs != target_crs:
    edges_plot = edges_plot.to_crs(target_crs)

# Try to find an existing Voronoi tessellation to draw (outer/full cells)
_voronoi_candidates = ["voronoi_gdf", "voro_gdf", "vor_gdf", "cells_gdf", "voronoi_cells",
                       "voronoi_full_gdf", "outer_voronoi_gdf", "vgrid_gdf"]
voronoi_gdf_to_plot = None
for _name in _voronoi_candidates:
    if _name in globals():
        _g = globals()[_name]
        if isinstance(_g, gpd.GeoDataFrame) and not _g.empty:
            voronoi_gdf_to_plot = _g
            break

fig, ax = plt.subplots(figsize=(14, 14))

# Set extent BEFORE basemap so tiles fetch the right area
if bound_plot is not None and not bound_plot.empty:
    bp = bound_plot if bound_plot.crs == target_crs else bound_plot.to_crs(target_crs)
    minx, miny, maxx, maxy = bp.total_bounds
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

# Add basemap using OSMnx wrapper with Mapnik only
provider_used = None
try:
    ox.plot.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=target_crs.to_string())
    provider_used = "osmnx:OpenStreetMap.Mapnik"
except Exception as e:
    try:
        ctx.add_basemap(ax, crs=target_crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zorder=-1)
        provider_used = "contextily:OpenStreetMap.Mapnik"
    except Exception as ee:
        print("[basemap] skipped after Mapnik failures:", e, ee)

# Draw order
if edges_plot is not None and not edges_plot.empty:
    edges_plot.plot(ax=ax, linewidth=0.35, alpha=0.6, color="#888888", zorder=1, label="_edges")

# Water (for reference)
if (R_KEY is not None) and ('water_by_radius' in globals()) and (R_KEY in water_by_radius) and not water_by_radius[R_KEY].empty:
    wplt = water_by_radius[R_KEY]
    if wplt.crs != target_crs: wplt = wplt.to_crs(target_crs)
    wplt.plot(ax=ax, color="steelblue", alpha=0.25, zorder=2, label="Water")

# Voronoi boundaries if available
if voronoi_gdf_to_plot is not None:
    vplt = voronoi_gdf_to_plot if voronoi_gdf_to_plot.crs == target_crs else voronoi_gdf_to_plot.to_crs(target_crs)
    vplt.boundary.plot(ax=ax, color="#555555", linewidth=0.8, alpha=0.9, zorder=3, label="Voronoi boundaries")

# Hub polygons:
#   - fill with the *land-clipped* inner polygons (inner_work)
#   - outline with the inner_gdf boundary
inner_work.plot(ax=ax, facecolor="orange", alpha=0.25, edgecolor="none", zorder=4)
inner_gdf.boundary.plot(ax=ax, color="#222222", linewidth=1.2, zorder=5, label="Hub (inner) boundary")

# Facilities
if not dwater_gdf.empty:
    dwater_gdf.plot(ax=ax, color="cyan", markersize=22, alpha=0.9, zorder=6, label="Drinking Water")
if not schools_gdf.empty:
    schools_gdf.plot(ax=ax, color="purple", markersize=22, alpha=0.9, zorder=6, label="Schools")
if not markets_gdf.empty:
    markets_gdf.plot(ax=ax, color="limegreen", markersize=22, alpha=0.9, zorder=6, label="Markets")
if not toilets_gdf.empty:
    toilets_gdf.plot(ax=ax, color="black", markersize=22, alpha=0.9, zorder=6, label="Public Toilets")

# boundary frame
if bound_plot is not None and not bound_plot.empty:
    bp.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=7, label="Boundary")

ax.set_title("Facilities allocated by hub node count + centrality (median-scaled)"
             + (f" — basemap: {provider_used}" if provider_used else ""), fontsize=15, pad=10)
ax.set_axis_off()

# Clean legend
handles, labels = ax.get_legend_handles_labels()
labels_clean, handles_clean = [], []
for h, l in zip(handles, labels):
    if l and not l.startswith("_"):
        labels_clean.append(l); handles_clean.append(h)
if labels_clean:
    ax.legend(handles_clean, labels_clean, loc="upper left", fontsize=9, frameon=True)

plt.tight_layout()
#plt.show()

# -------- summaries -----------------------------------------------------------
def _summary(df, name):
    if df.empty:
        return f"{name}: 0 (none)"
    counts = df.groupby("cluster").size().sort_index()
    lines = [f"{name} per cluster:"]
    for cid, cnt in counts.items():
        lines.append(f"  Cluster {int(cid)}: {int(cnt)}")
    lines.append(f"Total {name}: {int(counts.sum())}")
    return "\n".join(lines)

print(_summary(dwater_gdf,  "Drinking Water"))
print(_summary(schools_gdf, "Schools"))
print(_summary(markets_gdf, "Markets"))
print(_summary(toilets_gdf, "Public Toilets"))

# ===================== Notebook cell 24 =====================
# Install Mesa (Agent-Based Modeling Framework)

# ===================== Notebook cell 25 =====================
# === ABM CELL 1 — density-derived homes (global pop-weighted sampling, gradient preview) ===
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ─── A. HUMAN AGENT CLASS (unchanged) ────────────────────────────────────────
class HumanAgent:
    def __init__(self, unique_id, model, role, home_location, rho=0.5, gamma=0.7):
        self.unique_id = unique_id
        self.model = model
        self.role = role
        self.state = "S"
        self.days_in_state = 0
        self.hydration_level = max(0, np.random.normal(loc=80, scale=20))
        self.feces_timer     = max(1, int(np.random.normal(loc=8, scale=2)))
        self.exploration_prob  = rho
        self.exploration_decay = gamma
        self.visited_places = []
        self.home_location = home_location
        self.current_location = home_location
        self.home_buffer = None
        if random.random() < 0.01:
            self.state = "E"

    def step(self):
        self.hydration_level -= 30
        self.feces_timer -= 1
        if self.unique_id < 5:
            if self.hydration_level <= 20:
                self.hydration_level = 100.0
            if self.feces_timer <= 0:
                self.feces_timer = max(1, int(np.random.normal(loc=8, scale=2)))

# ─── B. PARAMETERS / INPUT LAYERS ────────────────────────────────────────────
assert 'outer_gdf' in globals() and not outer_gdf.empty, "outer_gdf missing (run Voronoi step)."
assert 'boundaries_by_radius' in globals() and 'R' in globals() and R in boundaries_by_radius, \
    "Current boundary not found. Run trimming cell first."

# Current boundary (at R), projected to target CRS
# (We use this as the *only* city boundary mask; no outer_gdf clipping for homes.)
bound_gdf = boundaries_by_radius[R]
target_crs = bound_gdf.crs if bound_gdf.crs is not None else outer_gdf.crs

if bound_gdf.crs != target_crs:
    bound_gdf = bound_gdf.to_crs(target_crs)
current_bound_union = unary_union(bound_gdf.geometry)

# Water layer (strictly from current R; no 7 km fallback)
if 'water_by_radius' in globals() and R in water_by_radius and not water_by_radius[R].empty:
    water_gdf = water_by_radius[R].copy()
elif 'water_gdf' in globals() and isinstance(water_gdf, gpd.GeoDataFrame):
    water_gdf = water_gdf.copy()
else:
    water_gdf = gpd.GeoDataFrame(geometry=[], crs=target_crs)

if water_gdf.crs is None:
    water_gdf = water_gdf.set_crs(target_crs)
elif water_gdf.crs != target_crs:
    water_gdf = water_gdf.to_crs(target_crs)

water_union = water_gdf.unary_union if not water_gdf.empty else Polygon()

# Global home sampling controls
area_per_home       = 2500      # m² per home (sets total target count over the whole city)
people_per_home     = 5
home_buffer_radius  = 0
RANDOM_SEED         = 42
rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Density field (KDE-like) controls for gradient + sampling
GRID_SPACING_M      = 200       # grid pitch for evaluating density
KERNEL_SIGMA_M      = 600       # Gaussian σ (meters)
KERNEL_CUTOFF_M     = 2_000     # ignore points beyond this when computing density
JITTER_M            = 60        # small jitter when placing homes on sampled grid cells

# ─── C. POP_XYZ LOADER (robust) ──────────────────────────────────────────────
def _load_pop_xyz(path_like, target_crs="EPSG:3857"):
    import pandas as pd
    try:
        df = pd.read_csv(path_like, engine="python")
    except Exception:
        df = pd.read_csv(path_like, delim_whitespace=True, header=None)

    if df.columns.dtype == 'O':
        cols_norm = [str(c).strip().lower() for c in df.columns]
    else:
        cols_norm = [f"col{i}" for i in range(len(df.columns))]

    name_map = {}
    for i, c in enumerate(cols_norm):
        if c in ("x","lon","longitude","easting"):   name_map[df.columns[i]] = "x"
        elif c in ("y","lat","latitude","northing"): name_map[df.columns[i]] = "y"
        elif c in ("z","pop","population","value"):  name_map[df.columns[i]] = "z"

    if {"x","y","z"}.issubset(set(name_map.values())):
        df = df.rename(columns=name_map)
    else:
        df = df.iloc[:, :3].copy()
        df.columns = ["x","y","z"]

    for c in ("x","y","z"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x","y","z"])

    g = gpd.GeoDataFrame(
        {"pop": df["z"].astype(float).values},
        geometry=gpd.points_from_xy(df["x"].astype(float).values,
                                    df["y"].astype(float).values),
        crs="EPSG:4326"
    )
    return g.to_crs(target_crs)

assert 'POP_XYZ' in globals(), "POP_XYZ path not defined (see PATHS cell)."
pop_pts = _load_pop_xyz(POP_XYZ, target_crs=target_crs)

# ─── D. CITY LAND = CURRENT R BOUNDARY MINUS WATER; CLIP POP TO LAND ─────────
# IMPORTANT CHANGE: no intersection with outer_gdf here.
city_mask = current_bound_union  # full R-boundary disk = "city" for home placement

# If MultiPolygon, keep the largest (small islands can skew the grid)
try:
    if city_mask.geom_type == "MultiPolygon":
        city_mask = max(city_mask.geoms, key=lambda p: p.area)
except Exception:
    pass

if not water_gdf.empty and not water_union.is_empty:
    land_poly = city_mask.difference(water_union)
else:
    land_poly = city_mask

# Clip population points to this land mask
pop_pts = pop_pts[pop_pts.geometry.within(land_poly)]

# ─── E. BUILD A GRID OVER LAND & EVALUATE A GAUSSIAN DENSITY FIELD ──────────
minx, miny, maxx, maxy = land_poly.bounds
gx = np.arange(minx, maxx, GRID_SPACING_M)
gy = np.arange(miny, maxy, GRID_SPACING_M)
GX, GY = np.meshgrid(gx, gy)
grid_xy = np.column_stack([GX.ravel(), GY.ravel()])

# keep only grid points that lie on land (inside R boundary and off-water)
grid_points = gpd.GeoSeries(gpd.points_from_xy(grid_xy[:,0], grid_xy[:,1]), crs=target_crs)
on_land_mask = grid_points.within(land_poly).values
grid_xy = grid_xy[on_land_mask]
grid_points = grid_points[on_land_mask]

# KDE-like density using cKDTree (Gaussian kernel)
if not pop_pts.empty:
    pts_xy = np.column_stack([pop_pts.geometry.x.values, pop_pts.geometry.y.values])
    weights = pop_pts["pop"].clip(lower=0).to_numpy().astype(float)
    tree = cKDTree(pts_xy)

    idxs_list = tree.query_ball_point(grid_xy, r=KERNEL_CUTOFF_M)
    inv_two_sigma2 = 1.0 / (2.0 * (KERNEL_SIGMA_M**2))
    dens = np.zeros(len(grid_xy), dtype=float)
    for i, neigh in enumerate(idxs_list):
        if not neigh:
            continue
        dx = pts_xy[neigh, 0] - grid_xy[i, 0]
        dy = pts_xy[neigh, 1] - grid_xy[i, 1]
        d2 = dx*dx + dy*dy
        k  = np.exp(-d2 * inv_two_sigma2)
        dens[i] = float((k * weights[neigh]).sum())
else:
    dens = np.ones(len(grid_xy), dtype=float)

# normalize to probabilities for sampling
dens = np.maximum(dens, 0)
if np.all(dens == 0) or not np.isfinite(dens).any():
    probs = np.full(len(dens), 1.0/len(dens))
else:
    probs = dens / dens.sum()

# ─── F. SAMPLE HOMES GLOBALLY (NOT PER OUTER CELL) ──────────────────────────
total_area = float(land_poly.area)
n_homes = max(1, int(total_area / area_per_home))

choice_idx = rng.choice(len(grid_xy), size=n_homes, replace=True, p=probs)
chosen_xy = grid_xy[choice_idx]

# jitter and ensure inside land (so never on water, never outside R)
homes = []
for x, y in chosen_xy:
    px = float(x) + rng.normal(0, JITTER_M)
    py = float(y) + rng.normal(0, JITTER_M)
    p  = Point(px, py)
    if not land_poly.contains(p):
        c = land_poly.representative_point()
        p = Point((p.x + c.x)/2, (p.y + c.y)/2)
        if not land_poly.contains(p):
            p = c
    homes.append(p)

homes_gdf = gpd.GeoDataFrame(
    {"people_count": [people_per_home]*len(homes)},
    geometry=homes,
    crs=target_crs
)
homes_gdf["home_buffer"] = homes_gdf.geometry.buffer(home_buffer_radius)

# ─── G. WATER CONTAMINATION REGISTRY (unchanged) ─────────────────────────────
if 'layers' in globals() and isinstance(layers, dict) and ("Water Bodies" in layers) and (layers["Water Bodies"] is not None):
    water_gdf_for_reg = layers["Water Bodies"]
elif 'water_bodies' in globals() and water_bodies is not None:
    water_gdf_for_reg = water_bodies
else:
    water_gdf_for_reg = water_gdf

if water_gdf_for_reg.crs != target_crs:
    water_gdf_for_reg = water_gdf_for_reg.to_crs(target_crs)

simplify_tolerance = 5
if not water_gdf_for_reg.empty:
    water_gdf_for_reg["geometry"] = water_gdf_for_reg.geometry.simplify(
        simplify_tolerance, preserve_topology=True
    )
    water_union_reg = unary_union(water_gdf_for_reg.geometry)
else:
    water_union_reg = Polygon([])

water_tree = STRtree(list(water_gdf_for_reg.geometry)) if not water_gdf_for_reg.empty else None
# Redesign: natural water remains spatial-only; tap registry is the epidemiological registry.
water_contamination = {}
tap_contamination = water_contamination
# Operational threshold in internal accumulated tap-load units; no water-volume
# conversion to CFU/ml is implemented.
threshold_per_m2 = 1e9

# ─── H. CONTAMINATE DRINKING WATER POIs (same heuristic) ─────────────────────
contaminated_pois = []
if 'drinking_points' in globals() and len(drinking_points) > 0:
    tap_points = list(drinking_points)
elif 'dwater_gdf' in globals() and isinstance(dwater_gdf, gpd.GeoDataFrame) and not dwater_gdf.empty:
    tap_points = list(dwater_gdf.geometry.values)
else:
    tap_points = []

for pt in tap_points:
    key = id(pt)
    if key not in water_contamination:
        water_contamination[key] = {
            "geometry": pt,
            "feces_load": 0,
            "area": 1,
            "threshold_total": threshold_per_m2 * 1.0,
            "contaminated": False
        }

if len(tap_points) > 0:
    k = max(1, int(float(globals().get("TAP_INITIAL_CONTAMINATION_FRAC", 0.10)) * len(tap_points)))
    contaminated_pois = random.sample(tap_points, k)
    for pt in contaminated_pois:
        rec = water_contamination[id(pt)]
        rec["feces_load"] = max(rec.get("feces_load", 0), 1e10)
        rec["contaminated"] = True
        rec["threshold_total"] = 1
else:
    print("[Contam] No drinking taps found yet — skipping seeding.")

water_contamination_by_wkb = {rec["geometry"].wkb: rec for rec in water_contamination.values()}
tap_contamination_by_wkb = water_contamination_by_wkb

# ─── I. GRADIENT PREVIEW + SUMMARY ───────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(8, 8))
    grad = gpd.GeoDataFrame({"dens": dens}, geometry=grid_points, crs=target_crs)
    land_gdf = gpd.GeoDataFrame(geometry=[land_poly], crs=target_crs)

    land_gdf.boundary.plot(
        ax=ax, color="black", linewidth=0.8, alpha=0.8,
        label="City boundary (R-boundary minus water)"
    )
    grad.plot(
        ax=ax, column="dens", cmap="viridis", markersize=10, alpha=0.6,
        legend=True, legend_kwds={"label": "Population gradient"}
    )
    if not water_gdf.empty:
        water_gdf.plot(ax=ax, color="steelblue", alpha=0.25, label="Water")

    if len(homes_gdf):
        homes_gdf.plot(
            ax=ax, markersize=6, alpha=0.9, color="#222222",
            label="Homes (pop-weighted)"
        )

    plt.title("Population gradient (color) and sampled homes (dots)")
    plt.axis("off")
    plt.legend(loc="upper right")
    #plt.show()
except Exception as e:
    print("[Visual] skipped:", e)

total_agents = int(homes_gdf["people_count"].sum()) if len(homes_gdf) else 0
print(f"Sampled {len(homes_gdf)} homes → {total_agents} agents (assuming {people_per_home} per home).")
print(f"{len(contaminated_pois)} drinking water POIs were pre-contaminated.")
print(f"{sum(1 for rec in water_contamination.values() if rec['contaminated'])} total contaminated water records.")

# ===================== Notebook cell 26 =====================
import random
import numpy as np
from collections import Counter

# ─── A. INSTANTIATE AGENTS FROM HOMES_GDF ────────────────────────────
agent_list = []
roles = ["student", "trader", "general"]

class DummyModel:
    pass

model = DummyModel()
uid = 0

# ─── Mobility heterogeneity (NEW): draw rho/gamma per agent from distributions ───
# Population means for exploration propensity and return decay.
# Population mean exploration parameter (dimensionless).
BASE_RHO_MEAN   = 0.5
# Population mean exploration-decay parameter (dimensionless).
BASE_GAMMA_MEAN = 0.7
# "Realistic" SDs (tunable): keep modest so behaviour stays close to baseline unless you change these.
RHO_SD   = 0.10
GAMMA_SD = 0.10

def _clip01(x):
    return max(0.0, min(1.0, float(x)))

def draw_rho_gamma(rho_mean=BASE_RHO_MEAN, gamma_mean=BASE_GAMMA_MEAN, rho_sd=RHO_SD, gamma_sd=GAMMA_SD):
    """Draw (rho, gamma) for an agent. rho is clipped to [0,1]; gamma is floored at 0.01."""
    rho   = _clip01(np.random.normal(loc=float(rho_mean),   scale=float(rho_sd)))
    gamma = max(0.01, float(np.random.normal(loc=float(gamma_mean), scale=float(gamma_sd))))
    return rho, gamma

for household_id, row in homes_gdf.iterrows():
    home_point  = row.geometry
    home_buffer = row["home_buffer"]
    people_cnt  = int(row["people_count"])

    house_agents = []
    for i in range(people_cnt):
        rho_i, gamma_i = draw_rho_gamma()
        ag = HumanAgent(
            unique_id=uid,
            model=model,
            role=random.choice(roles),
            home_location=home_point,
            rho=rho_i,
            gamma=gamma_i
        )
        # store baseline mobility params (so interventions can keep rho fixed)
        ag._rho_baseline   = rho_i
        ag._gamma_baseline = gamma_i

        # ─── Heterogeneous hydration + defecation ──────────────────────
        ag.hydration_level = max(0, np.random.normal(loc=35, scale=25))       # lower → more water-seeking
        ag.feces_timer     = max(1, int(np.random.normal(loc=8, scale=2)))    # more frequent defecation

        ag.home_buffer   = home_buffer
        ag.household_id  = household_id
        ag.immobile      = False
        house_agents.append(ag)
        agent_list.append(ag)
        uid += 1

    # ─── B. LESS IMMOBILE AGENTS FOR MORE MOVEMENT ─────────────────────
    num_immobile = max(0, people_cnt // 10)
    if num_immobile > 0:
        for ag in random.sample(house_agents, num_immobile):
            ag.immobile = True

# ─── C. SEED INITIAL EXPOSED AND INFECTED CASES ──────────────────────────
n_agents = len(agent_list)
initial_exposed_count  = min(int(float(globals().get("INITIAL_EXPOSED_FRAC", 0.01)) * n_agents), n_agents)
initial_infected_count = min(int(float(globals().get("INITIAL_INFECTED_FRAC", 0.02)) * n_agents), n_agents - initial_exposed_count)

initial_exposed  = random.sample(agent_list, initial_exposed_count)
remaining_agents = [ag for ag in agent_list if ag not in initial_exposed]
initial_infected = random.sample(remaining_agents, initial_infected_count)

for ag in initial_exposed:
    ag.state = "E"
    ag.days_in_state = 0

for ag in initial_infected:
    ag.state = "I"
    ag.days_in_state = 0

# ─── D. REPORT INITIAL STATE COUNTS ──────────────────────────────────
init_counts = Counter(ag.state for ag in agent_list)
print(f"Created {len(agent_list)} agents.")
print("Initial state distribution:")
for state in ["S", "E", "I", "R"]:
    print(f"  {state}: {init_counts.get(state, 0)} agents")

# ─── E. PRESERVE FULL ROSTER FOR FUTURE SAMPLING ─────────────────────
full_agent_list = agent_list.copy()

# ===================== Notebook cell 27 =====================
import matplotlib.pyplot as plt
import random
import geopandas as gpd
import contextily as ctx
import osmnx as ox
import numpy as np

# ─── A. SAMPLE AGENTS FOR VISUALIZATION ─────────────────────────────
if len(full_agent_list) == 0:
    raise RuntimeError("No agents found in full_agent_list.")
sampled_agents = random.sample(full_agent_list, min(10000, len(full_agent_list)))

# ─── B. DETERMINE HOUSEHOLD INFECTION STATUS ─────────────────────────
infected_households = {ag.household_id for ag in agent_list if ag.state == "I"}
exposed_households  = {ag.household_id for ag in agent_list if ag.state == "E"}

# ─── C. BUILD GEODATAFRAMES FOR PLOTTING ────────────────────────────
home_points  = [ag.home_location for ag in sampled_agents]
home_buffers = [ag.home_buffer    for ag in sampled_agents]

gdf_buffers = gpd.GeoDataFrame(geometry=home_buffers, crs=homes_gdf.crs)
gdf_points  = gpd.GeoDataFrame({
    "infected": [ag.household_id in infected_households for ag in sampled_agents],
    "exposed":  [ag.household_id in exposed_households  for ag in sampled_agents]
}, geometry=home_points, crs=homes_gdf.crs)

# ─── D. PICK BOUNDARY (strictly use current R boundary) & harmonize CRS ─────
bound_plot = None
if "boundaries_by_radius" in globals():
    if "R" in globals() and R in boundaries_by_radius:
        bound_plot = boundaries_by_radius[R]
    elif "R_GLOBAL_M" in globals() and int(R_GLOBAL_M) in boundaries_by_radius:
        bound_plot = boundaries_by_radius[int(R_GLOBAL_M)]

target_crs = (bound_plot.crs if (bound_plot is not None and hasattr(bound_plot, "crs") and bound_plot.crs)
              else gdf_points.crs)
gdf_buffers = gdf_buffers.to_crs(target_crs)
gdf_points  = gdf_points.to_crs(target_crs)

# ─── E. POPULATION DENSITY LAYER (better gradient) ───────────────────
pop_for_plot = None
if "pop_pts" in globals() and isinstance(pop_pts, gpd.GeoDataFrame) and not pop_pts.empty:
    pop_for_plot = pop_pts if pop_pts.crs == target_crs else pop_pts.to_crs(target_crs)

# ─── F. WATER LAYER ──────────────────────────────────────────────────
water_map = None
if "water_gdf" in globals() and isinstance(water_gdf, gpd.GeoDataFrame) and not water_gdf.empty:
    water_map = water_gdf if water_gdf.crs == target_crs else water_gdf.to_crs(target_crs)
    if "water_contamination_by_wkb" in globals():
        wc_wkb = water_contamination_by_wkb
        water_map["contaminated"] = water_map.geometry.apply(lambda g: wc_wkb.get(g.wkb, {}).get("contaminated", False))
    elif "water_contamination" in globals():
        water_map["contaminated"] = [water_contamination.get(id(geom), {}).get("contaminated", False)
                                     for geom in water_map.geometry]
    else:
        water_map["contaminated"] = False

# ─── G. PLOT ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 12))

# Set extent BEFORE basemap
if bound_plot is not None and not bound_plot.empty:
    bp = bound_plot if bound_plot.crs == target_crs else bound_plot.to_crs(target_crs)
    minx, miny, maxx, maxy = bp.total_bounds
else:
    minx, miny, maxx, maxy = gdf_points.total_bounds
ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

# Basemap
try:
    ctx.add_basemap(ax, crs=target_crs, source=ctx.providers.OpenStreetMap.Mapnik)
except Exception:
    pass

# (1) Population density gradient as background (hexbin) — HIGH CONTRAST
from matplotlib.colors import LogNorm, PowerNorm

if pop_for_plot is not None and not pop_for_plot.empty:
    x = pop_for_plot.geometry.x.values
    y = pop_for_plot.geometry.y.values
    c = np.asarray(pop_for_plot["pop"].values, dtype=float)

    # ---- contrast controls ----
    STRETCH   = "log"      # "log", "power", "linear"
    P_LOW     = 2.0
    P_HIGH    = 99.5
    GAMMA     = 0.45
    GRID_SIZE = 110

    p_lo = np.percentile(c, P_LOW)
    p_hi = np.percentile(c, P_HIGH)
    c = np.clip(c, p_lo, p_hi)

    if STRETCH == "log":
        vmin = max(1e-9, p_lo)
        vmax = max(vmin * 10, p_hi)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif STRETCH == "power":
        norm = PowerNorm(gamma=GAMMA, vmin=p_lo, vmax=p_hi)
    else:
        norm = None

    hb = ax.hexbin(
        x, y,
        C=c,
        reduce_C_function=np.sum,
        gridsize=GRID_SIZE,
        mincnt=1,
        linewidths=0,
        alpha=0.9,
        cmap="magma",
        norm=norm,
        zorder=1
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.030, pad=0.01)
    cb.set_label("Population density (hexbin sum, contrast-stretched)")

# (2) Water
if water_map is not None and not water_map.empty:
    water_map.plot(
        ax=ax,
        color=water_map["contaminated"].map({False: "lightblue", True: "red"}),
        alpha=0.35, edgecolor="black", linewidth=0.4,
        label="Water", zorder=2
    )

# (2b) Voronoi + inner polygons (BOUNDARY OVERLAYS — THICK & VISIBLE)
outer_plot = None
if "outer_gdf" in globals() and isinstance(outer_gdf, gpd.GeoDataFrame) and not outer_gdf.empty:
    outer_plot = outer_gdf if outer_gdf.crs == target_crs else outer_gdf.to_crs(target_crs)

    # black halo for visibility
    outer_plot.boundary.plot(
        ax=ax,
        color="black",
        linewidth=3.5,
        alpha=0.9,
        zorder=3
    )
    # colored line on top
    outer_plot.boundary.plot(
        ax=ax,
        color="red",
        linewidth=2.2,
        alpha=1.0,
        zorder=3.2,
        label="Outer Voronoi"
    )

inner_plot = None
if "inner_gdf" in globals() and isinstance(inner_gdf, gpd.GeoDataFrame) and not inner_gdf.empty:
    inner_plot = inner_gdf if inner_gdf.crs == target_crs else inner_gdf.to_crs(target_crs)

    # black halo
    inner_plot.boundary.plot(
        ax=ax,
        color="black",
        linewidth=4.0,
        alpha=0.9,
        zorder=3.3
    )
    # yellow line on top
    inner_plot.boundary.plot(
        ax=ax,
        color="yellow",
        linewidth=2.8,
        alpha=1.0,
        zorder=3.5,
        label="Inner polygons"
    )


# (3) Optional: draw fewer healthy points for clarity
healthy_mask = ~gdf_points["infected"] & ~gdf_points["exposed"]
if healthy_mask.sum() > 3000:
    healthy_idx = np.random.choice(np.where(healthy_mask)[0], 3000, replace=False)
    healthy_sub = gdf_points.iloc[healthy_idx]
else:
    healthy_sub = gdf_points[healthy_mask]

# (4) Plot by infection status
healthy_sub.plot(ax=ax, color="green", markersize=4, label="Healthy Households", zorder=4)
gdf_points[gdf_points["exposed"] & ~gdf_points["infected"]].plot(
    ax=ax, color="orange", markersize=6, label="Exposed Households", zorder=5)
gdf_points[gdf_points["infected"]].plot(
    ax=ax, color="red", markersize=6, label="Infectious Households", zorder=6)

# (5) Boundary (R boundary)
if bound_plot is not None and not bound_plot.empty:
    bp.boundary.plot(ax=ax, color='black', linewidth=1, zorder=7)

# Legend & title
ax.set_title("Homes over Population Gradient", fontsize=15)
ax.axis("off")
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), loc="upper right", frameon=True)

plt.tight_layout()
# ─── SAVE FIGURE (like your R5 example) ─────────────────────────────
assert 'CITY_SLUG' in globals() and 'CITY_OUT' in globals(), "Run R0a first to set CITY_SLUG and CITY_OUT."
os.makedirs(CITY_OUT, exist_ok=True)
out_png = os.path.join(CITY_OUT, f"{CITY_SLUG}_homes_population_gradient.png")
plt.savefig(OUT_DIR /out_png, dpi=300, bbox_inches="tight")
#plt.show()
#print(f"[Fig.homes] Saved {out_png}")

# ===================== Notebook cell 28 =====================
# ─── DEFINE CORE CHOLERA ABM MODEL ─────────────────────────────────────
class CholeraModel:
    def __init__(self, agents):
        self.agents = agents
        self.current_tick = 0
        self.total_agents = len(agents)
        print(f"Model initialized with {self.total_agents} agents.")

    def step(self):
        # Step 1: Advance all agents
        for agent in self.agents:
            agent.step()

        # Step 2: Optional - Household transmission
        self._apply_household_exposure()

        self.current_tick += 1

    def _apply_household_exposure(self):
        """
        If any agent in a household is Infectious (I),
        all Susceptible (S) household members become Exposed (E).
        (Same behavior as before; probability logic—if any—is handled elsewhere.)
        """
        household_groups = {}
        for ag in self.agents:
            household_groups.setdefault(ag.household_id, []).append(ag)

        p_hh = float(globals().get("HOUSEHOLD_SECONDARY_EXPOSURE_PROB", 0.08))
        for agents in household_groups.values():
            infectious_present = any(ag.state == "I" for ag in agents)
            if infectious_present:
                for ag in agents:
                    if ag.state == "S" and random.random() < p_hh:   # probabilistic household secondary exposure
                        ag.state = "E"
                        ag.days_in_state = 0

# ─── INSTANTIATE MODEL ────────────────────────────────────────────────
cholera_model = CholeraModel(full_agent_list)

# ─── ADVANCE ONE STEP (OPTIONAL) ─────────────────────────────────────
cholera_model.step()

# ===================== Notebook cell 29 =====================
# ───────────────────────────────────────────────────────────────────────────────
# SEIRS STEP FUNCTION + TOILET/WATER INFRASTRUCTURE + CONTAMINATION LOGIC
# Optimized with STRtree indices and prebuilt household map for speed
# ───────────────────────────────────────────────────────────────────────────────

from shapely.ops import nearest_points
from shapely.strtree import STRtree
import geopandas as gpd
import random
import numpy as np
from shapely.geometry import Point


def _items_empty(items):
    """Safe emptiness test for lists/Series/GeoSeries/GeoDataFrames."""
    if items is None:
        return True
    try:
        if hasattr(items, "empty") and items.empty:
            return True
    except Exception:
        pass
    try:
        return len(items) == 0
    except Exception:
        return False

# ─── 0. PRE-SEED: INITIAL EXPOSED (1%) & INFECTED (30%) ─────────────────────────
num_agents      = len(full_agent_list)
num_exposed     = max(1, int(float(globals().get("INITIAL_EXPOSED_FRAC", 0.01)) * num_agents))
exposed_agents  = random.sample(full_agent_list, num_exposed)
remaining       = [ag for ag in full_agent_list if ag not in exposed_agents]
num_infected    = max(1, int(float(globals().get("INITIAL_INFECTED_FRAC", 0.02)) * num_agents))
infected_agents = random.sample(remaining, num_infected)

for ag in exposed_agents:
    ag.state         = "E"
    ag.days_in_state = 0
for ag in infected_agents:
    ag.state         = "I"
    ag.days_in_state = 0

# ─── 1. SETUP: Toilets & Water Contamination Infrastructure ────────────────

# (CHANGED) Public toilets — prefer synthetic toilets_gdf; fallback to layers if missing
if "toilets_gdf" in globals() and isinstance(toilets_gdf, gpd.GeoDataFrame):
    toilet_gdf = toilets_gdf.copy()
elif "layers" in globals() and isinstance(layers.get("Public Toilets"), gpd.GeoDataFrame):
    toilet_gdf = layers["Public Toilets"].copy()
else:
    toilet_gdf = gpd.GeoDataFrame(geometry=[], crs=water_gdf.crs if "water_gdf" in globals() else "EPSG:3857")

toilet_geoms  = list(toilet_gdf.geometry) if not toilet_gdf.empty else []
# (kept) usage map; we’ll key by WKB when we increment to avoid id-mismatch on copies
toilet_usage  = {getattr(t, "wkb", None): 0 for t in toilet_geoms}
max_toilet_load = 10  # ← use your capacity of 10
toilet_tree   = STRtree(toilet_geoms) if len(toilet_geoms) else None  # guard

# (UNCHANGED SOURCE) Drinking-water taps (from synthetic facility cell)
tap_points = list(drinking_points) if ('drinking_points' in globals() and len(drinking_points)) else []
tap_tree   = STRtree(tap_points) if len(tap_points) else None  # guard

# Inner‐zone polygons
inner_polys = list(inner_gdf.geometry) if ('inner_gdf' in globals() and not inner_gdf.empty) else []
inner_tree  = STRtree(inner_polys) if len(inner_polys) else None  # guard

# Natural water bodies remain spatial-only masks in the redesigned logic.
water_geoms = list(water_gdf.geometry)
water_tree  = STRtree(water_geoms) if len(water_geoms) else None
threshold_per_m2   = 1e9

# Tap-only contamination registry (kept backward-compatible via water_contamination aliases).
if 'tap_contamination' not in globals():
    tap_contamination = {}
if 'drinking_points' in globals():
    for pt in drinking_points:
        key = id(pt)
        if key not in tap_contamination:
            tap_contamination[key] = {
                "geometry":        pt,
                "feces_load":      0,
                "area":            1.0,
                "threshold_total": threshold_per_m2 * 1.0,
                "contaminated":    False
            }

tap_contamination_by_wkb = {rec["geometry"].wkb: rec for rec in tap_contamination.values()}
water_contamination = tap_contamination
water_contamination_by_wkb = tap_contamination_by_wkb

def _ensure_tap_registry():
    """Make sure every tap has a record in the tap-only contamination registry."""
    if 'drinking_points' not in globals():
        return
    for pt in drinking_points:
        key = id(pt)
        if key not in tap_contamination:
            tap_contamination[key] = {
                "geometry":        pt,
                "feces_load":      0,
                "area":            1.0,
                "threshold_total": threshold_per_m2 * 1.0,
                "contaminated":    False
            }

def _propagate_tap_contamination_from_water(water_geoms_in):
    """Disabled in tap-only redesign: natural water is spatial-only."""
    return

# Random home-water contamination (final-paper global fraction)
contaminated_homes = set(random.sample(list(homes_gdf.index), max(1, int(float(globals().get("HOME_INITIAL_CONTAMINATION_FRAC", 0.012)) * len(homes_gdf)))))
home_contamination = {hid: {"contaminated": (hid in contaminated_homes)} for hid in homes_gdf.index.tolist()}

# Prebuild household → agents map for fast “household‐wide exposure”
household_map = {}
for ag in full_agent_list:
    household_map.setdefault(ag.household_id, []).append(ag)

# ─── Shapely 2.x STRtree adapters (indices → geometries) ──────────────────────
def _tree_query_geoms(tree, items, target_geom):
    if tree is None or _items_empty(items):
        return []
    res = tree.query(target_geom)
    if res is None:
        return []
    try:
        seq = list(res)
    except TypeError:
        seq = [res]
    if not seq:
        return []
    if isinstance(seq[0], (np.integer, int)):
        return [items[int(i)] for i in seq]
    return seq

def _tree_nearest_geom(tree, items, ref_pt):
    if tree is None or _items_empty(items):
        return None
    res = tree.nearest(ref_pt)
    if isinstance(res, (np.integer, int)):
        return items[int(res)]
    if isinstance(res, tuple) and len(res) >= 1:
        g = res[0]
        if isinstance(g, (np.integer, int)):
            return items[int(g)]
        return g
    return res

# ─── 2. Helper: Feces CFU per disease state ─────────────────────────────────
def get_feces_cfu(state):
    return {"S": 100, "E": 150, "I": 2_000_000_000, "R": 0}[state]


# --- diagnostics-only exposure source tracker (does not alter model logic) ---
CURRENT_TICK_EXPOSURE_SOURCE = {}

def _record_exposure_source(agent_or_uid, source):
    """Diagnostics only: remember the first inferred source for an agent becoming E this tick."""
    try:
        uid = int(agent_or_uid.unique_id)
    except Exception:
        uid = int(agent_or_uid)
    if uid not in CURRENT_TICK_EXPOSURE_SOURCE:
        CURRENT_TICK_EXPOSURE_SOURCE[uid] = str(source)

# ─── 3. Helper: Spatial Utilities ─────────────────────────────────────────
def get_inner_polygon(location):
    if inner_tree is None:
        return None
    candidates = _tree_query_geoms(inner_tree, inner_polys, location)
    for p in candidates:
        try:
            if p.contains(location):
                return p
        except Exception:
            continue
    return None

HOME_USE_RADIUS_M = 25.0

def _is_at_home_point(loc, home_loc):
    try:
        if loc is None or home_loc is None:
            return False
        if not isinstance(loc, Point) and hasattr(loc, "centroid"):
            loc = loc.centroid
        if not isinstance(home_loc, Point) and hasattr(home_loc, "centroid"):
            home_loc = home_loc.centroid
        return float(loc.distance(home_loc)) <= float(globals().get("HOME_USE_RADIUS_M", 25.0))
    except Exception:
        return False

def is_near_contaminated_water(loc, max_dist=2000):
    return False

# ─── 4. Main SEIRS Agent Step Function ───────────────────────────────────────
infection_risk            = 1.0
exposed_to_infectious     = 5 * 24
infectious_to_recovered   = 10 * 24
# Recovered-state duration (hours); exceeds the simulated seven-day horizon.
recovered_to_susceptible  = 100 * 24

def seirs_step(self):
    loc = self.current_location

    # —— A) HOUSEHOLD‐WIDE EXPOSURE (instant) ——  
    if self.state == "I":
        p_hh = float(globals().get("HOUSEHOLD_SECONDARY_EXPOSURE_PROB", 0.08))
        for ag2 in household_map[self.household_id]:
            if ag2.state == "S" and random.random() < p_hh:
                ag2.state         = "E"
                ag2.days_in_state = 0

    # —— B) DISEASE CLOCK PROGRESSION ——  
    self.hydration_level -= 30
    self.feces_timer     -= 1
    self.days_in_state   += 1

    if   self.state == "E" and self.days_in_state >= exposed_to_infectious:
        self.state = "I"; self.days_in_state = 0
    elif self.state == "I" and self.days_in_state >= infectious_to_recovered:
        self.state = "R"; self.days_in_state = 0
    elif self.state == "R" and self.days_in_state >= recovered_to_susceptible:
        self.state = "S"; self.days_in_state = 0

    # —— C) DRINKING: Home OR Nearest Drinking‐Water Tap when thirsty ——  
    if self.hydration_level <= 20:
        drank       = False
        source_home = False

        # 1) If at home, drink at home
        if _is_at_home_point(loc, self.home_location):
            self.current_location = self.home_location
            source_home = True
            drank       = True

        # 2) Otherwise, nearest tap in inner zone or globally
        else:
            inner = get_inner_polygon(loc)
            if inner is not None and tap_tree is not None:
                taps_here = _tree_query_geoms(tap_tree, tap_points, inner)
                taps_here = [pt for pt in taps_here if inner.contains(pt)]
                if taps_here:
                    pt = min(taps_here, key=lambda p: p.distance(loc))
                    self.current_location = pt
                    drank = True

            if not drank and tap_tree is not None:
                try:
                    pt = _tree_nearest_geom(tap_tree, tap_points, loc)
                    if isinstance(pt, Point):
                        self.current_location = pt
                        drank = True
                except Exception:
                    pass

        # Reset hydration
        if drank:
            self.hydration_level = 100.0

        # Instant exposure if source is contaminated
        if drank and self.state == "S":
            home_risk = source_home and (self.household_id in contaminated_homes)

            # robust contamination lookup
            rec = tap_contamination.get(id(self.current_location)) if 'tap_contamination' in globals() else None
            if rec is None:
                try:
                    rec = tap_contamination_by_wkb.get(self.current_location.wkb) if 'tap_contamination_by_wkb' in globals() else None
                except Exception:
                    rec = None
            tap_contaminated = bool(rec and rec.get("contaminated", False))
            tap_risk = tap_contaminated and (random.random() < float(globals().get("TAP_EXPOSURE_PROB", 0.15)))

            if home_risk or tap_risk:
                self.state         = "E"
                self.days_in_state = 0
                _record_exposure_source(self, "home_water" if home_risk else "tap_water")

    # —— D) DEFECATION: Home → Nearest Toilet → Water ——  
    if self.feces_timer <= 0:
        # A) If at home, defecate at home
        if _is_at_home_point(loc, self.home_location):
            self.feces_timer = max(1, int(np.random.normal(loc=8, scale=2)))
            return

        # B) Otherwise, nearest toilet if under capacity
        if len(toilet_geoms) and toilet_tree is not None:
            try:
                nearest_toilet = _tree_nearest_geom(toilet_tree, toilet_geoms, loc)
            except Exception:
                nearest_toilet = None
            if nearest_toilet is not None:
                key_wkb = getattr(nearest_toilet, "wkb", None)
                load = toilet_usage.get(key_wkb, 0)
                if load < max_toilet_load:
                    self.current_location = nearest_toilet
                    toilet_usage[key_wkb] = load + 1
                    self.feces_timer = max(1, int(np.random.normal(loc=8, scale=2)))
                    return

        # C) If toilet full or none, open-defecate nearby and load the nearest drinking tap.
        nearest_tap = _tree_nearest_geom(tap_tree, tap_points, loc) if tap_tree is not None else None
        if nearest_tap is not None:
            key_t = id(nearest_tap)
            if 'tap_contamination' not in globals():
                globals()['tap_contamination'] = {}
            if key_t not in tap_contamination:
                tap_contamination[key_t] = {
                    "geometry":        nearest_tap,
                    "feces_load":      0,
                    "area":            1.0,
                    "threshold_total": threshold_per_m2 * 1.0,
                    "contaminated":    False
                }
            self.current_location = loc
            tap_contamination[key_t]["feces_load"] += get_feces_cfu(self.state)
            globals()['water_contamination'] = tap_contamination
            globals()['tap_contamination_by_wkb'] = {rec["geometry"].wkb: rec for rec in tap_contamination.values()}
            globals()['water_contamination_by_wkb'] = globals()['tap_contamination_by_wkb']

        # D) Reset feces timer
        self.feces_timer = max(1, int(np.random.normal(loc=8, scale=2)))

# ─── 5. BIND TO ALL AGENTS ───────────────────────────────────────────────
for agent in full_agent_list:
    agent.step = seirs_step.__get__(agent, HumanAgent)

# ─── 6. MARK WATER BODIES CONTAMINATED WHEN THRESHOLD EXCEEDED ───────────
def update_water_contamination(current_tick=None):
    registry = globals().get("tap_contamination", globals().get("water_contamination", {}))
    for rec in registry.values():
        if not rec["contaminated"] and rec["feces_load"] >= rec["threshold_total"]:
            # Existing model mechanism: crossing the threshold switches the tap
            # to the same binary contaminated state used everywhere else.
            rec["contaminated"] = True
            # Record whether contamination was acquired during simulation.
            # threshold crossing, and when it first occurred.
            if not bool(rec.get("initially_contaminated", False)):
                rec["dynamically_contaminated"] = True
                if rec.get("contamination_tick", None) is None and current_tick is not None:
                    rec["contamination_tick"] = int(current_tick)
    globals()["tap_contamination"] = registry
    globals()["water_contamination"] = registry
    globals()["tap_contamination_by_wkb"] = {rec["geometry"].wkb: rec for rec in registry.values()}
    globals()["water_contamination_by_wkb"] = globals()["tap_contamination_by_wkb"]

# ===================== Notebook cell 30 =====================
# Advance the model manually by stepping each agent and updating contamination

# Example: Simulate a single tick (e.g., 1 hour)
for agent in full_agent_list:
    agent.step()

# Update contamination state of water bodies
update_water_contamination()

# ===================== Notebook cell 31 =====================
# ─── AGENT MOBILITY, DESTINATION CHOICE & INTEGRATED (COMPOSED) STEP ──────

import random
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

# ─────────────────────────────────────────────────────────────────────────
# 1) BUILD DESTINATION POOLS
#    - Schools/Markets from synthetic facilities you generated earlier
#    - "dEPR"/general destinations = high-centrality street nodes INSIDE inner hubs
# ─────────────────────────────────────────────────────────────────────────

# A) Synthetic facilities (robust if any are missing)
school_points = list(schools_gdf.geometry) if ("schools_gdf" in globals() and not schools_gdf.empty) else []
market_points = list(markets_gdf.geometry) if ("markets_gdf" in globals() and not markets_gdf.empty) else []

# B) High-centrality nodes citywide (agents do not need to stay inside inner polygons)
#    Prefer density-derived global graph (R_GLOBAL_M), else fall back to 7 km.
def _mobility_nodes_source_key():
    if "centrality_by_radius" not in globals():
        return None
    if "R_GLOBAL_M" in globals() and int(R_GLOBAL_M) in centrality_by_radius:
        return int(R_GLOBAL_M)
    if 7_000 in centrality_by_radius:
        return 7_000
    return None

def _centrality_score_array(nodes_gdf):
    g = nodes_gdf.get("C_global", pd.Series(0.0, index=nodes_gdf.index)).astype(float).to_numpy()
    l = nodes_gdf.get("C_local",  pd.Series(0.0, index=nodes_gdf.index)).astype(float).to_numpy()
    score = np.maximum(g, l).astype(float)
    score[~np.isfinite(score)] = 0.0
    return score

def _nearest_node_scores_for_points(points, nodes_gdf):
    if not points or nodes_gdf is None or getattr(nodes_gdf, "empty", True):
        return np.zeros(len(points), dtype=float)
    try:
        node_xy = np.column_stack((nodes_gdf.geometry.x.to_numpy(dtype=float), nodes_gdf.geometry.y.to_numpy(dtype=float)))
        pt_xy = np.array([[float(p.x), float(p.y)] for p in points], dtype=float)
        if len(node_xy) == 0 or len(pt_xy) == 0:
            return np.zeros(len(points), dtype=float)
        tree = cKDTree(node_xy)
        _, idx = tree.query(pt_xy, k=1)
        score = _centrality_score_array(nodes_gdf)
        return score[np.asarray(idx, dtype=int)]
    except Exception:
        return np.ones(len(points), dtype=float)

def _build_weighted_mobility_pool(generic_points, generic_weights, school_pts, market_pts, nodes_gdf):
    generic_points = list(generic_points) if generic_points is not None else []
    school_pts = list(school_pts) if school_pts is not None else []
    market_pts = list(market_pts) if market_pts is not None else []

    if generic_weights is None:
        gw = np.ones(len(generic_points), dtype=float)
    else:
        try:
            gw = np.asarray(generic_weights, dtype=float).copy()
        except Exception:
            gw = np.ones(len(generic_points), dtype=float)
    if gw.size != len(generic_points):
        gw = np.ones(len(generic_points), dtype=float)
    gw[~np.isfinite(gw)] = 0.0

    sw = _nearest_node_scores_for_points(school_pts, nodes_gdf)
    mw = _nearest_node_scores_for_points(market_pts, nodes_gdf)

    points = list(generic_points) + list(school_pts) + list(market_pts)
    weights = np.concatenate([gw, sw, mw]).astype(float) if len(points) else np.asarray([], dtype=float)
    if weights.size:
        weights[~np.isfinite(weights)] = 0.0
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights, dtype=float)
        weights = weights / float(weights.sum())
    return points, weights

depr_points = []
depr_weights = None
MOBILITY_POINTS = []
MOBILITY_WEIGHTS = None

nodes_src_key = _mobility_nodes_source_key()
_mobility_nodes_gdf = None
if nodes_src_key is not None:
    _mobility_nodes_gdf = centrality_by_radius[nodes_src_key].copy()
    target_crs = inner_gdf.crs if ('inner_gdf' in globals() and inner_gdf is not None and not inner_gdf.empty) else _mobility_nodes_gdf.crs
    if hasattr(_mobility_nodes_gdf, "crs") and _mobility_nodes_gdf.crs != target_crs:
        try:
            _mobility_nodes_gdf = _mobility_nodes_gdf.to_crs(target_crs)
        except Exception:
            pass
    if "boundaries_by_radius" in globals() and nodes_src_key in boundaries_by_radius:
        try:
            _bound = boundaries_by_radius[nodes_src_key]
            if getattr(_bound, 'crs', None) != target_crs:
                _bound = _bound.to_crs(target_crs)
            _mobility_nodes_gdf = gpd.clip(_mobility_nodes_gdf, _bound)
        except Exception:
            pass

    if _mobility_nodes_gdf is not None and not _mobility_nodes_gdf.empty:
        score = _centrality_score_array(_mobility_nodes_gdf)
        thr = float(np.nanpercentile(score, 80.0)) if len(score) else 0.0
        keep = score >= thr
        nodes_in = _mobility_nodes_gdf.loc[keep].copy()
        score = score[keep]
        if not nodes_in.empty:
            depr_points = list(nodes_in.geometry.values)
            s = np.asarray(score, dtype=float)
            s[~np.isfinite(s)] = 0.0
            total = float(s.sum())
            depr_weights = (s / total) if total > 0 else np.full(len(depr_points), 1.0 / max(1, len(depr_points)), dtype=float)

if depr_points:
    MOBILITY_POINTS, MOBILITY_WEIGHTS = _build_weighted_mobility_pool(depr_points, depr_weights, school_points, market_points, _mobility_nodes_gdf)
    depr_points = list(MOBILITY_POINTS)
    depr_weights = None if MOBILITY_WEIGHTS is None else np.asarray(MOBILITY_WEIGHTS, dtype=float)

# Fallback if we couldn’t build a pool (should be rare)
if not depr_points:
    depr_points = []
    depr_weights = None
    MOBILITY_POINTS = []
    MOBILITY_WEIGHTS = None

# ─────────────────────────────────────────────────────────────────────────
# 2) GRAVITY-STYLE DESTINATION CHOICE (supports centrality weights)
#    - Exploration probability decays with #visited (your rho, gamma live on agents)
#    - For revisits, multiply by visit frequency
#    - Always distance-decay; optional prior weights (centrality) if provided
# ─────────────────────────────────────────────────────────────────────────
def select_destination(agent, poi_list, alpha=2.0, eps=1e-6, base_weights=None):
    # Optional intervention cap on movement radius (meters)
    max_r = globals().get('MOVE_MAX_RADIUS_M', None)

    if not poi_list:
        return agent.current_location  # nowhere to go

    S = max(1, len(agent.visited_places))
    P_new = getattr(agent, "exploration_prob", 0.5) * (S ** -getattr(agent, "exploration_decay", 0.7))
    exploring = random.random() < P_new

    # Candidate set
    if base_weights is not None:
        if (not hasattr(select_destination, "_w_cache")
                or select_destination._w_cache is not poi_list
                or len(getattr(select_destination, "_w_index", {})) != len(poi_list)):
            select_destination._w_cache = poi_list
            select_destination._w_index = {id(geom): i for i, geom in enumerate(poi_list)}

    if exploring:
        candidates = [p for p in poi_list if p not in agent.visited_places] or poi_list
    else:
        candidates = agent.visited_places or poi_list

    w = None
    if base_weights is not None:
        idxs = [select_destination._w_index.get(id(p), None) for p in candidates]
        w = np.array([base_weights[i] if i is not None else 0.0 for i in idxs], dtype=float)

    # Current location sanity
    curr = agent.current_location
    if not isinstance(curr, Point):
        curr = curr.centroid
        agent.current_location = curr

    # Enforce intervention-specific movement radius cap if active.
    if max_r is not None:
        try:
            max_r = float(max_r)
        except Exception:
            max_r = None
    if max_r is not None and np.isfinite(max_r) and max_r > 0:
        dist_all = np.array([curr.distance(p) for p in candidates], dtype=float)
        within_mask = dist_all <= max_r
        if np.any(within_mask):
            candidates = [p for p, keep in zip(candidates, within_mask) if keep]
            if w is not None:
                w = w[within_mask]
        else:
            return min(candidates, key=lambda p: curr.distance(p))

    Rs = np.array([curr.distance(p) for p in candidates], dtype=float)
    # Use kilometres for gravity distance decay so 1/r^alpha is not over-localised by metre units.
    Rs_km = np.maximum(Rs / 1000.0, 0.25)
    d_term = 1.0 / (Rs_km**alpha + eps)

    if w is not None:
        d_term = d_term * np.maximum(w, eps)

    if not exploring:
        freq = np.array([agent.visited_places.count(p) for p in candidates], dtype=float)
        d_term = d_term * np.maximum(freq, 1.0)

    total = d_term.sum()
    if total <= 0 or not np.isfinite(total):
        return min(candidates, key=lambda p: curr.distance(p))
    probs = d_term / total
    idx = np.random.choice(len(candidates), p=probs)
    return candidates[idx]

# ─────────────────────────────────────────────────────────────────────────
# 3) ROLE/TIME MOBILITY POLICY
# ─────────────────────────────────────────────────────────────────────────
def mobility_logic(agent, tick_hour, step_size=5000):
    if getattr(agent, "immobile", False):
        return

    if not isinstance(agent.current_location, Point):
        agent.current_location = agent.current_location.centroid

    if agent.hydration_level <= 20 or agent.feces_timer <= 0:
        if random.random() > 0.90:
            return

    if tick_hour >= 23 or tick_hour < 5:
        agent.destination = agent.home_location if random.random() < 0.8 else \
            select_destination(agent, depr_points, base_weights=depr_weights)

    elif 9 <= tick_hour < 15 and getattr(agent, "role", "general") == "student":
        if school_points and random.random() < 0.25:
            _school_w = _nearest_node_scores_for_points(school_points, _mobility_nodes_gdf) if '_mobility_nodes_gdf' in globals() else None
            agent.destination = select_destination(agent, school_points, base_weights=_school_w)
        else:
            agent.destination = select_destination(agent, depr_points, base_weights=depr_weights)

    elif 9 <= tick_hour < 15 and getattr(agent, "role", "general") == "trader":
        if market_points and random.random() < 0.48:
            _market_w = _nearest_node_scores_for_points(market_points, _mobility_nodes_gdf) if '_mobility_nodes_gdf' in globals() else None
            agent.destination = select_destination(agent, market_points, base_weights=_market_w)
        else:
            agent.destination = select_destination(agent, depr_points, base_weights=depr_weights)

    else:
        agent.destination = select_destination(agent, depr_points, base_weights=depr_weights)

    dest = agent.destination if agent.destination is not None else agent.current_location
    cur  = agent.current_location
    dx, dy = dest.x - cur.x, dest.y - cur.y
    dist = math.hypot(dx, dy)

    if dist <= step_size:
        new_loc = dest
    else:
        r = step_size / dist
        new_loc = Point(cur.x + dx * r, cur.y + dy * r)

    agent.current_location = new_loc

    if new_loc.equals(dest) and dest not in agent.visited_places:
        agent.visited_places.append(dest)

# ─────────────────────────────────────────────────────────────────────────
# 4) COMPOSE MOBILITY WITH YOUR EXISTING STEP (no overwrite)
# ─────────────────────────────────────────────────────────────────────────
for ag in full_agent_list:
    prev_step = ag.step  # whatever you bound earlier (SEIRS + contamination, etc.)

    def composed_step(self, _prev=prev_step):
        _prev()
        hour = getattr(self.model, "current_tick", 0) % 24
        mobility_logic(self, hour)

    ag.step = composed_step.__get__(ag, type(ag))

# ===================== Notebook cell 33 =====================
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.ops import unary_union

# ─── A. Setup Contamination Map (tap-only epidemiology; water stays spatial only) ─────
# Operational threshold in internal accumulated tap-load units; no water-volume
# conversion to CFU/ml is implemented.
threshold_per_m2 = 1e9

if 'tap_contamination' not in globals():
    tap_contamination = {}

# Ensure all drinking taps are present (keep existing flags)
if 'drinking_points' in globals():
    for pt in drinking_points:
        key = id(pt)
        if key not in tap_contamination:
            tap_contamination[key] = {
                "geometry":        pt,
                "feces_load":      0,
                "area":            1.0,
                "threshold_total": threshold_per_m2 * 1.0,
                "contaminated":    False
            }

# Build/refresh WKB mirrors (lookup helper + backward compatibility)
tap_contamination_by_wkb = {
    rec["geometry"].wkb: rec for rec in tap_contamination.values()
}
water_contamination = tap_contamination
water_contamination_by_wkb = tap_contamination_by_wkb

# --- Water → Tap spill rule disabled in tap-only redesign ---
def _ensure_tap_registry_vec():
    if 'drinking_points' in globals():
        for pt in drinking_points:
            key = id(pt)
            if key not in tap_contamination:
                tap_contamination[key] = {
                    "geometry":        pt,
                    "feces_load":      0,
                    "area":            1.0,
                    "threshold_total": threshold_per_m2 * 1.0,
                    "contaminated":    False
                }

def _propagate_tap_contamination_from_water_vec(water_geoms):
    return

# ─── B. Feces CFU per state (unchanged) ───────────────────────────────────────
def get_feces_cfu(state):
    # State-specific contributions in the internal tap-loading scheme.
    feces_map = {"S": 100, "E": 150, "I": 2_000_000_000, "R": 0}
    return feces_map[state]

# ─── C. Update Water Contamination Flags (unchanged) ──────────────────────────
def update_water_contamination(current_tick=None):
    registry = globals().get("tap_contamination", globals().get("water_contamination", {}))
    for rec in registry.values():
        if not rec["contaminated"] and rec["feces_load"] >= rec["threshold_total"]:
            # Existing model mechanism: crossing the threshold switches the tap
            # to the same binary contaminated state used everywhere else.
            rec["contaminated"] = True
            # Record the origin and time of the contamination transition.
            if not bool(rec.get("initially_contaminated", False)):
                rec["dynamically_contaminated"] = True
                if rec.get("contamination_tick", None) is None and current_tick is not None:
                    rec["contamination_tick"] = int(current_tick)
    globals()["tap_contamination"] = registry
    globals()["water_contamination"] = registry
    globals()["tap_contamination_by_wkb"] = {rec["geometry"].wkb: rec for rec in registry.values()}
    globals()["water_contamination_by_wkb"] = globals()["tap_contamination_by_wkb"]

# ─── D. Trees & helpers for inner-first facility selection (new targets only) ─
# taps (synthetic drinking points)
tap_points = drinking_points if 'drinking_points' in globals() else []
tap_tree   = STRtree(tap_points) if len(tap_points) else None

# toilets (synthetic)
toilet_gdf = toilets_gdf.copy() if 'toilets_gdf' in globals() else gpd.GeoDataFrame(geometry=[], crs=water_gdf.crs)
toilet_geoms = list(toilet_gdf.geometry) if not toilet_gdf.empty else []
toilet_tree  = STRtree(toilet_geoms) if len(toilet_geoms) else None
MAX_TOILET_LOAD = 10  # ← your threshold of 10

# natural water retained as spatial-only layer/mask
water_geoms = list(water_gdf.geometry)
water_tree  = STRtree(water_geoms) if len(water_geoms) else None

# inner polygons
inner_polys = list(inner_gdf.geometry) if 'inner_gdf' in globals() else []
inner_tree  = STRtree(inner_polys) if len(inner_polys) else None

# --- Adapters for Shapely 2.x STRtree returning indices OR geometries -----------
def _tree_query_geoms(tree, items, target_geom):
    """Return a list of geometry objects for candidates intersecting target_geom.

    NOTE: items can be a list/tuple of geometries OR a GeoSeries/GeoDataFrame.
    """
    if tree is None or items is None:
        return []

    # Handle GeoDataFrame/GeoSeries/pandas Series safely
    try:
        if hasattr(items, "empty") and items.empty:
            return []
    except Exception:
        pass

    # Convert items into an indexable 1D geometry sequence
    if "geopandas" in globals():
        try:
            import geopandas as gpd
            if isinstance(items, gpd.GeoDataFrame):
                items_seq = items.geometry
            elif isinstance(items, gpd.GeoSeries):
                items_seq = items
            else:
                items_seq = items
        except Exception:
            items_seq = items
    else:
        items_seq = items

    # If still a DataFrame-like, attempt geometry attribute
    if hasattr(items_seq, "geometry") and not hasattr(items_seq, "iloc"):
        try:
            items_seq = items_seq.geometry
        except Exception:
            pass

    # If it's list-like with len
    try:
        if hasattr(items_seq, "__len__") and len(items_seq) == 0:
            return []
    except Exception:
        pass

    cands = tree.query(target_geom)
    if cands is None:
        return []

    # Shapely may return a numpy array; cast to list
    try:
        seq = list(cands)
    except TypeError:
        seq = [cands]
    if not seq:
        return []

    # If they are indices, map to items; else they are geometries
    if isinstance(seq[0], (np.integer, int)):
        out = []
        for i in seq:
            try:
                out.append(items_seq.iloc[int(i)] if hasattr(items_seq, "iloc") else items_seq[int(i)])
            except Exception:
                pass
        return out
    else:
        return seq

def _find_inner(poly_tree, pt):
    cands = _tree_query_geoms(poly_tree, inner_polys, pt)
    for p in cands:
        try:
            if p.contains(pt):
                return p
        except Exception:
            continue
    return None

def _nearest_in_polygon(tree, items, poly, ref_pt):
    if tree is None or _items_empty(items) or poly is None:
        return None
    cands = _tree_query_geoms(tree, items, poly)
    cands = [g for g in cands if hasattr(g, "within") and poly.contains(g)]
    if not cands:
        return None
    return min(cands, key=lambda g: ref_pt.distance(g))

def _nearest_global(tree, items, ref_pt):
    return _tree_nearest_geom(tree, items, ref_pt)

# --- tiny sanitizer to guarantee we only assign Points back into the GeoDataFrame
def _as_point(val, fallback):
    if isinstance(val, Point):
        return val
    try:
        if hasattr(val, "centroid"):
            c = val.centroid
            if isinstance(c, Point):
                return c
    except Exception:
        pass
    return fallback

# ─── E. Vectorised simulation-step function ─────
def run_vectorized_step(agent_df):
    # 1. SEIRS & Resource Progression
    agent_df["hydration_level"] -= 30
    agent_df["feces_timer"]     -= 1
    agent_df["days_in_state"]   += 1

    # 2. SEIRS Transitions (unchanged)
    agent_df.loc[(agent_df["state"] == "E") & (agent_df["days_in_state"] >= 5 * 24), ["state", "days_in_state"]] = ["I", 0]
    agent_df.loc[(agent_df["state"] == "I") & (agent_df["days_in_state"] >= 10 * 24), ["state", "days_in_state"]] = ["R", 0]
    agent_df.loc[(agent_df["state"] == "R") & (agent_df["days_in_state"] >= 100 * 24), ["state", "days_in_state"]] = ["S", 0]

    _ensure_tap_registry_vec()
    globals()["water_contamination"] = globals().get("tap_contamination", {})
    globals()["water_contamination_by_wkb"] = globals().get("tap_contamination_by_wkb", {})

    # 3. DRINKING: home if at home, otherwise taps only (inner-first)
    need_water = agent_df["hydration_level"] <= 20
    agent_df["_drank_source_home"] = False
    agent_df["_drank_tap_wkb"] = None
    agent_df["_destination_type"] = "none"

    if need_water.any():
        for idx, row in agent_df.loc[need_water].iterrows():
            cur = row["current_location"]
            if not isinstance(cur, Point) and hasattr(cur, "centroid"):
                cur = cur.centroid

            home_loc = row.get("home_location", cur)
            at_home = _is_at_home_point(cur, home_loc)

            if at_home:
                home_loc = row.get("home_location", cur)
                agent_df.at[idx, "current_location"] = _as_point(home_loc, cur)
                agent_df.at[idx, "_drank_source_home"] = True
                agent_df.at[idx, "_destination_type"] = "home"
            else:
                inner = _find_inner(inner_tree, cur)
                tap = None
                if inner is not None:
                    tap = _nearest_in_polygon(tap_tree, tap_points, inner, cur)
                if tap is None:
                    tap = _nearest_global(tap_tree, tap_points, cur)
                if tap is not None:
                    tap_pt = _as_point(tap, cur)
                    agent_df.at[idx, "current_location"] = tap_pt
                    agent_df.at[idx, "_destination_type"] = "tap"
                    try:
                        agent_df.at[idx, "_drank_tap_wkb"] = tap_pt.wkb
                    except Exception:
                        agent_df.at[idx, "_drank_tap_wkb"] = None
                else:
                    agent_df.at[idx, "current_location"] = cur
                    agent_df.at[idx, "_destination_type"] = "none"
                    agent_df.at[idx, "_drank_tap_wkb"] = None

        agent_df.loc[need_water, "hydration_level"] = 100.0

    # 4. DEFECATION: home -> toilet if capacity allows -> open defecation nearby loads nearest tap
    need_defecate = agent_df["feces_timer"] <= 0
    if need_defecate.any():
        agent_df.loc[need_defecate, "feces_timer"] = np.maximum(
            1,
            np.random.normal(loc=8, scale=2, size=need_defecate.sum()).astype(int)
        )

        idx_def = agent_df.index[need_defecate]
        at_home_mask = agent_df.loc[idx_def].apply(
            lambda r: _is_at_home_point(r.get("current_location", None), r.get("home_location", None)),
            axis=1
        ).values if "home_location" in agent_df.columns else np.zeros(len(idx_def), dtype=bool)

        idx_nonhome = idx_def[~at_home_mask]
        if len(idx_nonhome) > 0:
            toilet_usage = {getattr(t, "wkb", None): 0 for t in toilet_geoms}
            new_locs_nonhome = []

            for i in idx_nonhome:
                row = agent_df.loc[i]
                cur = row["current_location"]
                if not isinstance(cur, Point) and hasattr(cur, "centroid"):
                    cur = cur.centroid

                inner = _find_inner(inner_tree, cur)
                tlt = None
                if inner is not None:
                    tlt = _nearest_in_polygon(toilet_tree, toilet_geoms, inner, cur)
                if tlt is None:
                    tlt = _nearest_global(toilet_tree, toilet_geoms, cur)

                used_toilet = False
                if tlt is not None:
                    k = getattr(tlt, "wkb", None)
                    load = toilet_usage.get(k, 0)
                    if load < MAX_TOILET_LOAD:
                        new_locs_nonhome.append((i, tlt))
                        agent_df.at[i, "_destination_type"] = "toilet"
                        toilet_usage[k] = load + 1
                        used_toilet = True

                if not used_toilet:
                    nearest_tap = _nearest_global(tap_tree, tap_points, cur)
                    if nearest_tap is not None:
                        key_t = id(nearest_tap)
                        if key_t not in tap_contamination:
                            tap_contamination[key_t] = {
                                "geometry": nearest_tap,
                                "feces_load": 0,
                                "area": 1.0,
                                "threshold_total": threshold_per_m2 * 1.0,
                                "contaminated": False,
                            }
                        tap_contamination[key_t]["feces_load"] += get_feces_cfu(row["state"])
                    new_locs_nonhome.append((i, cur))

            if new_locs_nonhome:
                idxs, locs = zip(*new_locs_nonhome)
                locs_sanitized = [
                    _as_point(loc, agent_df.loc[i, "current_location"]) for i, loc in new_locs_nonhome
                ]
                agent_df.loc[list(idxs), "current_location"] = locs_sanitized

    # 5. Drinking contamination exposure: only home contamination or actual consumed tap contamination
    drank = need_water
    at_risk = drank & (agent_df["state"] == "S")
    for idx in agent_df.index[at_risk]:
        exposed = False
        source = None

        if bool(agent_df.at[idx, "_drank_source_home"]):
            hid = int(agent_df.at[idx, "household_id"])
            exposed = bool(globals().get("home_contamination", {}).get(hid, {}).get("contaminated", False))
            source = "home_water"
        else:
            tap_wkb = agent_df.at[idx, "_drank_tap_wkb"]
            if tap_wkb is not None:
                rec = globals().get("tap_contamination_by_wkb", {}).get(tap_wkb)
                tap_contaminated = bool(rec and rec.get("contaminated", False))
                exposed = tap_contaminated and (np.random.random() < float(globals().get("TAP_EXPOSURE_PROB", 0.15)))
                source = "tap_water"

        if exposed:
            agent_df.loc[idx, ["state", "days_in_state"]] = ["E", 0]
            _record_exposure_source(agent_df.loc[idx, "unique_id"], source or "tap_water")

    # 6. Household-Wide Exposure (probabilistic; prevents automatic 100% household conversion)
    infected_households = set(agent_df.loc[agent_df["state"] == "I", "household_id"])
    hh_mask = (agent_df["state"] == "S") & agent_df["household_id"].isin(infected_households)
    if hh_mask.any():
        p_hh = float(globals().get("HOUSEHOLD_SECONDARY_EXPOSURE_PROB", 0.08))
        draw = np.random.random(int(hh_mask.sum())) < p_hh
        newly_hh = agent_df.index[hh_mask][draw]
    else:
        newly_hh = agent_df.index[[]]
    agent_df.loc[newly_hh, ["state", "days_in_state"]] = ["E", 0]
    for _uid in newly_hh.tolist():
        _record_exposure_source(agent_df.loc[_uid, "unique_id"], "household_secondary")

    globals()["tap_contamination"] = tap_contamination
    globals()["water_contamination"] = tap_contamination
    globals()["tap_contamination_by_wkb"] = {rec["geometry"].wkb: rec for rec in tap_contamination.values()}
    globals()["water_contamination_by_wkb"] = globals()["tap_contamination_by_wkb"]

    return agent_df

# ===================== Notebook cell 34 =====================
# ─── BUILD POI POOLS (NO EXTERNAL POI DATA) ───────────────────────────────────
# Uses: inner_gdf, drinking_points, toilets_gdf, centrality_by_radius[7000]
# Output: POI_POOLS dict + a generic high-centrality POI list for movement

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# CRS we’ll use everywhere
_target_crs = inner_gdf.crs

# 1) High-centrality nodes citywide → generic exploration POIs
#    We’ll use the 80th-percentile of the same centrality score used by the mobility model.
_high_c_points = []
_high_c_weights = None
_mobility_nodes_gdf_pool = None
_nodes_pool_key = _mobility_nodes_source_key() if '_mobility_nodes_source_key' in globals() else None
if _nodes_pool_key is not None and "centrality_by_radius" in globals() and _nodes_pool_key in centrality_by_radius:
    _mobility_nodes_gdf_pool = centrality_by_radius[_nodes_pool_key].copy()
    if _mobility_nodes_gdf_pool.crs != _target_crs:
        _mobility_nodes_gdf_pool = _mobility_nodes_gdf_pool.to_crs(_target_crs)
    if "boundaries_by_radius" in globals() and _nodes_pool_key in boundaries_by_radius:
        try:
            _bound_pool = boundaries_by_radius[_nodes_pool_key]
            if _bound_pool.crs != _target_crs:
                _bound_pool = _bound_pool.to_crs(_target_crs)
            _mobility_nodes_gdf_pool = gpd.clip(_mobility_nodes_gdf_pool, _bound_pool)
        except Exception:
            pass
    if not _mobility_nodes_gdf_pool.empty and "C_global" in _mobility_nodes_gdf_pool.columns:
        _score_pool = _centrality_score_array(_mobility_nodes_gdf_pool) if '_centrality_score_array' in globals() else _mobility_nodes_gdf_pool["C_global"].astype(float).to_numpy()
        thr = float(np.nanpercentile(_score_pool, 80.0)) if len(_score_pool) else 0.0
        _keep_pool = _score_pool >= thr
        _nodes_core = _mobility_nodes_gdf_pool.loc[_keep_pool].copy()
        _score_core = np.asarray(_score_pool[_keep_pool], dtype=float)
        if not _nodes_core.empty:
            _high_c_points = list(_nodes_core.geometry)
            _score_core[~np.isfinite(_score_core)] = 0.0
            _sum_core = float(_score_core.sum())
            _high_c_weights = (_score_core / _sum_core) if _sum_core > 0 else np.full(len(_high_c_points), 1.0 / max(1, len(_high_c_points)), dtype=float)

# Fallbacks if needed
if not _high_c_points:
    _high_c_points = [poly.centroid for poly in inner_gdf.geometry]
    _high_c_weights = np.full(len(_high_c_points), 1.0 / max(1, len(_high_c_points)), dtype=float)

# 2) Synthetic facility POIs (already created earlier)
_poi_taps    = list(drinking_points) if 'drinking_points' in globals() else []
_poi_toilets = list(toilets_gdf.geometry) if 'toilets_gdf' in globals() and not toilets_gdf.empty else []
_poi_schools = list(schools_gdf.geometry) if 'schools_gdf' in globals() and not schools_gdf.empty else []
_poi_markets = list(markets_gdf.geometry) if 'markets_gdf' in globals() and not markets_gdf.empty else []

MOBILITY_POINTS, MOBILITY_WEIGHTS = _build_weighted_mobility_pool(_high_c_points, _high_c_weights, _poi_schools, _poi_markets, _mobility_nodes_gdf_pool) if '_build_weighted_mobility_pool' in globals() else (list(_high_c_points), np.asarray(_high_c_weights, dtype=float) if _high_c_weights is not None else None)
depr_points = list(MOBILITY_POINTS)
depr_weights = None if MOBILITY_WEIGHTS is None else np.asarray(MOBILITY_WEIGHTS, dtype=float)

# 3) Assemble pools
POI_POOLS = {
    "generic": list(MOBILITY_POINTS),
    "school":  _poi_schools,
    "market":  _poi_markets,
    "taps":    _poi_taps,
    "toilets": _poi_toilets
}

# Quick counts so you can sanity-check
print("[POI POOLS]")
for k, v in POI_POOLS.items():
    print(f"  {k}: {len(v)}")

# ===================== Notebook cell 35 =====================
from shapely.geometry import Point
import numpy as np

def add_movement_to_vectorized_step(agent_df, step_size=5000):
    """
    Move each agent one step toward its destination.
    If no destination or already arrived, assign a new one from POI pools.
    POI pools are expected to be defined globally as POI_POOLS with keys:
    'school', 'market', and 'generic'.
    """
    # Prepare next destinations
    new_destinations = []
    for idx, row in agent_df.iterrows():
        cur = row["current_location"]
        dest = row.get("destination", None)

        if not isinstance(dest, Point) or cur.equals(dest):
            role = row.get("role", "general")

            if role == "student" and "school" in POI_POOLS and POI_POOLS["school"]:
                dest = np.random.choice(POI_POOLS["school"])
            elif role == "trader" and "market" in POI_POOLS and POI_POOLS["market"]:
                dest = np.random.choice(POI_POOLS["market"])
            elif "generic" in POI_POOLS and POI_POOLS["generic"]:
                dest = np.random.choice(POI_POOLS["generic"])
            else:
                dest = cur  # fallback

        new_destinations.append(dest)

    agent_df["destination"] = new_destinations

    # Compute next location (bounded by step_size)
    new_locations = []
    for cur, dest in zip(agent_df["current_location"], agent_df["destination"]):
        dx = dest.x - cur.x
        dy = dest.y - cur.y
        dist = np.hypot(dx, dy)
        if dist <= step_size:
            new_loc = dest
        else:
            ratio = step_size / dist
            new_loc = Point(cur.x + dx * ratio, cur.y + dy * ratio)
        new_locations.append(new_loc)

    agent_df["current_location"] = new_locations
    return agent_df

# ===================== Notebook cell 36 =====================
# ══════════════════════════════════════════════════════════════
# QUICK MOVEMENT CHECK — Sanity check if agents are actually moving
# ══════════════════════════════════════════════════════════════

from random import sample

# Simulate a specific hour (e.g., 0 = midnight)
cholera_model.current_tick = 0

# Sample up to 100 agents for testing
n_check = min(100, len(agent_list))
sample_agents = sample(agent_list, n_check)

# Record initial positions
initial_positions = {ag.unique_id: ag.current_location for ag in sample_agents}

# Perform one step
for ag in sample_agents:
    ag.step()

# Compare positions before and after
moved = [
    ag for ag in sample_agents
    if ag.current_location != initial_positions.get(ag.unique_id)
]

print(f"{len(moved)} / {n_check} sampled agents moved.")

# Show a few examples
for ag in moved[:5]:
    before = initial_positions[ag.unique_id]
    after = ag.current_location
    print(f"Agent {ag.unique_id}: {before} → {after}")

if len(moved) == 0:
    print(" No agents moved. Check mobility logic or hydration/feces block conditions.")

# ===================== Notebook cell 37 =====================
# ══════════════════════════════════════════════════════════════
# CHECK SYNTHETIC FACILITIES AND HUB DESTINATIONS
# ══════════════════════════════════════════════════════════════

def _count_safe(gdf_name):
    g = globals().get(gdf_name)
    return 0 if g is None or g.empty else len(g)

n_water   = _count_safe("dwater_gdf")
n_schools = _count_safe("schools_gdf")
n_markets = _count_safe("markets_gdf")
n_toilets = _count_safe("toilets_gdf")

# Check high-centrality nodes pool (for generic/hub movement)
if "centrality_by_radius" in globals() and 7_000 in centrality_by_radius:
    nodes7 = centrality_by_radius[7_000]
    if {"C_global_z", "C_local_z"}.issubset(nodes7.columns):
        high_nodes = nodes7.query("(C_global_z >= 1) or (C_local_z >= 1)")
        n_high = len(high_nodes)
    else:
        n_high = len(nodes7)
else:
    n_high = 0

print("Synthetic Facilities and Hub Nodes Available for Movement")
print("--------------------------------------------------------")
print(f"Drinking water points : {n_water}")
print(f"Schools               : {n_schools}")
print(f"Markets               : {n_markets}")
print(f"Public Toilets        : {n_toilets}")
print(f"High-centrality nodes : {n_high}")

if (n_water + n_schools + n_markets + n_toilets + n_high) == 0:
    print("⚠️ No movement destinations defined! Agents will remain static.")
else:
    print("✅ Facility and hub destination pools are ready.")

# ===================== Notebook cell 38 =====================
# R0b — RUN CONFIG & HELPERS (rest of old R0 moved AFTER Zipf)
# Uses CITY_NAME / CITY_SLUG / SHARED_OUT / CITY_OUT already defined in R0a.

import os, random, numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from collections import defaultdict, Counter

# ── Core sim/runtime params ─────────────────────────────────────────────
TICKS_PER_DAY = globals().get("TICKS_PER_DAY", 24)
RUN_SEED  = globals().get("RUN_SEED", 42)
RUN_TICKS = globals().get("RUN_TICKS", 168)   # 7 days = 168 hours/ticks

# Epidemiological calibration constants used consistently by baseline and interventions.
# Incidence is new infections only (S→E); E→I is disease progression, not a new infection.
# Final-paper setting: ALL normal scenarios and ALL sensitivity/Sobol scenarios use
# the same epidemiological constants so differences are attributable to mobility
# and/or decentralisation mechanisms rather than hidden transmission-probability changes.
INITIAL_EXPOSED_FRAC = 0.01
INITIAL_INFECTED_FRAC = 0.02
TAP_INITIAL_CONTAMINATION_FRAC = 0.10
HOME_INITIAL_CONTAMINATION_FRAC = 0.012
TAP_EXPOSURE_PROB = 0.15
HOUSEHOLD_SECONDARY_EXPOSURE_PROB = 0.08

FINAL_EPI_PARAMS = {
    "INITIAL_EXPOSED_FRAC": INITIAL_EXPOSED_FRAC,
    "INITIAL_INFECTED_FRAC": INITIAL_INFECTED_FRAC,
    "TAP_INITIAL_CONTAMINATION_FRAC": TAP_INITIAL_CONTAMINATION_FRAC,
    "HOME_INITIAL_CONTAMINATION_FRAC": HOME_INITIAL_CONTAMINATION_FRAC,
    "TAP_EXPOSURE_PROB": TAP_EXPOSURE_PROB,
    "HOUSEHOLD_SECONDARY_EXPOSURE_PROB": HOUSEHOLD_SECONDARY_EXPOSURE_PROB,
}

def _scenario_epi_overrides(intervention):
    """Return fixed epidemiological parameters for every scenario.

    Final-paper logic:
    - Baseline, B, D1, D2, D3, grid sensitivity, and Sobol all use the
      same initial exposed/infected fractions and exposure probabilities.
    - Therefore intervention effects come from movement/decentralisation logic
      only, not from scenario-specific reductions in transmission risk.
    - ``preserve_baseline_epi=True`` is still added to intervention dictionaries
      as an explicit safety flag, but the helper itself no longer applies
      B/D-specific epidemiology overrides.

    Explicit epidemiological keys in an intervention dictionary are ignored on
    purpose here, so accidental per-scenario disease-risk changes cannot slip
    into final-paper runs.
    """
    return dict(FINAL_EPI_PARAMS)

# ── Ensure output dirs exist ───────────────────────────────────────────
assert 'SHARED_OUT' in globals() and 'CITY_OUT' in globals(), \
    "Run R0a first to set SHARED_OUT and CITY_OUT."
os.makedirs(SHARED_OUT, exist_ok=True)
os.makedirs(CITY_OUT,   exist_ok=True)

print(f"[R0b] CITY_NAME = {globals().get('CITY_NAME','(unknown)')}")
print(f"[R0b] SHARED_OUT = {SHARED_OUT} | CITY_OUT = {CITY_OUT}")
print(f"[R0b] TICKS_PER_DAY = {TICKS_PER_DAY} | RUN_SEED = {RUN_SEED} | RUN_TICKS = {RUN_TICKS}")

# ── Utilities ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def agents_to_gdf(agent_list):
    df = pd.DataFrame({
        "unique_id":       [a.unique_id for a in agent_list],
        "household_id":    [a.household_id for a in agent_list],
        "role":            [getattr(a, "role", "general") for a in agent_list],
        "state":           [a.state for a in agent_list],
        "days_in_state":   [a.days_in_state for a in agent_list],
        "hydration_level": [getattr(a, "hydration_level", 100) for a in agent_list],
        "feces_timer":     [getattr(a, "feces_timer", 8) for a in agent_list],
        "home_location":   [getattr(a, "home_location", a.current_location) for a in agent_list],
        "home_buffer":     [getattr(a, "home_buffer", None) for a in agent_list],
        "destination":     [getattr(a, "destination", None) for a in agent_list],
        "current_location":[a.current_location for a in agent_list],
    })
    return gpd.GeoDataFrame(df, geometry="current_location", crs="EPSG:3857")

# NEW: daily incidence percentage (/day)
def daily_incidence_percent(hist_df, ticks_per_day=TICKS_PER_DAY):
    N = int(hist_df.iloc[0][["S","E","I","R"]].sum())
    S = hist_df["S"].to_numpy()
    E = hist_df["E"].to_numpy()
    dSE = np.maximum(0, -np.diff(np.r_[S[0], S]))  # S→E only; E→I is progression
    inc_tick_pct = 100.0 * dSE / max(N, 1)
    day_index = np.arange(len(inc_tick_pct)) // ticks_per_day
    daily_pct = pd.Series(inc_tick_pct).groupby(day_index).sum()
    daily_pct.index.name = "day"
    daily_pct.name = "inc"
    return daily_pct

def active_infectious_percent(hist_df, ticks_per_day=TICKS_PER_DAY):
    """Daily mean active infectious prevalence (% I).

    This is deliberately separate from incidence: incidence is new S→E events,
    while prevalence is the current share of agents in state I. Spatial hotspot
    maps based on I should be compared to this prevalence series, not to S→E
    incidence.
    """
    if hist_df is None or hist_df.empty or "I" not in hist_df.columns:
        return pd.Series(dtype=float, name="active_infectious_pct")
    if {"S", "E", "I", "R"}.issubset(hist_df.columns):
        N = int(hist_df.iloc[0][["S", "E", "I", "R"]].sum())
    else:
        N = max(int(hist_df["I"].max()), 1)
    prev_tick_pct = 100.0 * hist_df["I"].astype(float).to_numpy() / max(N, 1)
    day_index = np.arange(len(prev_tick_pct)) // ticks_per_day
    daily_prev = pd.Series(prev_tick_pct).groupby(day_index).mean()
    daily_prev.index.name = "day"
    daily_prev.name = "active_infectious_pct"
    return daily_prev


def six_hour_incidence_percent(hist_df, bin_ticks=6):
    """6-hour binned incidence (% of population), using S decrease as S→E events."""
    cols = ["tick_start", "tick_end", "tick_mid", "incidence_pct"]
    if hist_df is None or hist_df.empty or not {"S", "E", "I", "R"}.issubset(hist_df.columns):
        return pd.DataFrame(columns=cols)
    N = int(hist_df.iloc[0][["S", "E", "I", "R"]].sum())
    S = hist_df["S"].astype(float).to_numpy()
    dSE = np.maximum(0.0, -np.diff(np.r_[S[0], S]))
    inc_tick_pct = 100.0 * dSE / max(N, 1)
    tick = np.arange(len(inc_tick_pct), dtype=int)
    bin_ticks = max(int(bin_ticks), 1)
    bin_id = tick // bin_ticks
    out = pd.DataFrame({"bin_id": bin_id, "tick": tick, "incidence_pct": inc_tick_pct})
    out = out.groupby("bin_id", as_index=False).agg(
        tick_start=("tick", "min"),
        tick_end=("tick", "max"),
        incidence_pct=("incidence_pct", "sum"),
    )
    out["tick_mid"] = (out["tick_start"] + out["tick_end"]) / 2.0
    return out[cols]


def six_hour_incidence_percent_from_transitions(trans_df, bin_ticks=6):
    """6-hour binned incidence (% of population), using logged S→E transition counts.

    This is the preferred incidence calculation for scenario/seed plots because it
    uses the transition ledger created inside run_scenario rather than inferring
    incidence from changes in the S compartment.
    """
    cols = ["tick_start", "tick_end", "tick_mid", "incidence_pct"]
    if trans_df is None or getattr(trans_df, "empty", True) or not {"tick", "SE", "N"}.issubset(trans_df.columns):
        return pd.DataFrame(columns=cols)
    df = trans_df[["tick", "SE", "N"]].copy()
    df["tick"] = df["tick"].astype(int)
    N0 = int(df["N"].dropna().iloc[0]) if df["N"].notna().any() else 1
    df["incidence_pct"] = 100.0 * df["SE"].astype(float) / max(N0, 1)
    bin_ticks = max(int(bin_ticks), 1)
    df["bin_id"] = df["tick"] // bin_ticks
    out = df.groupby("bin_id", as_index=False).agg(
        tick_start=("tick", "min"),
        tick_end=("tick", "max"),
        incidence_pct=("incidence_pct", "sum"),
    )
    out["tick_mid"] = (out["tick_start"] + out["tick_end"]) / 2.0
    return out[cols]

def daily_incidence_percent_from_transitions(trans_df, ticks_per_day=TICKS_PER_DAY):
    """Daily incidence (% of population), using logged S→E transition counts."""
    if trans_df is None or getattr(trans_df, "empty", True) or not {"tick", "SE", "N"}.issubset(trans_df.columns):
        return pd.Series(dtype=float, name="inc")
    df = trans_df[["tick", "SE", "N"]].copy()
    N0 = int(df["N"].dropna().iloc[0]) if df["N"].notna().any() else 1
    df["day"] = df["tick"].astype(int) // int(ticks_per_day)
    df["inc"] = 100.0 * df["SE"].astype(float) / max(N0, 1)
    out = df.groupby("day")["inc"].sum()
    out.index.name = "day"
    out.name = "inc"
    return out


def six_hour_active_infectious_percent(hist_df, bin_ticks=6):
    """6-hour binned active infectious prevalence (% I), averaged within each bin."""
    cols = ["tick_start", "tick_end", "tick_mid", "active_infectious_pct"]
    if hist_df is None or hist_df.empty or "I" not in hist_df.columns:
        return pd.DataFrame(columns=cols)
    if {"S", "E", "I", "R"}.issubset(hist_df.columns):
        N = int(hist_df.iloc[0][["S", "E", "I", "R"]].sum())
    else:
        N = max(int(hist_df["I"].max()), 1)
    prev_tick_pct = 100.0 * hist_df["I"].astype(float).to_numpy() / max(N, 1)
    tick = np.arange(len(prev_tick_pct), dtype=int)
    bin_ticks = max(int(bin_ticks), 1)
    bin_id = tick // bin_ticks
    out = pd.DataFrame({"bin_id": bin_id, "tick": tick, "active_infectious_pct": prev_tick_pct})
    out = out.groupby("bin_id", as_index=False).agg(
        tick_start=("tick", "min"),
        tick_end=("tick", "max"),
        active_infectious_pct=("active_infectious_pct", "mean"),
    )
    out["tick_mid"] = (out["tick_start"] + out["tick_end"]) / 2.0
    return out[cols]

# Establish destination order for POI pools
def freeze_iterable_orders():
    """Sorts geometry lists/tables so RNG draws are reproducible across runs."""
    global drinking_points, dwater_gdf, POI_POOLS, MOBILITY_POINTS, MOBILITY_WEIGHTS

    if 'drinking_points' in globals() and drinking_points:
        drinking_points = sorted(
            [pt for pt in drinking_points if pt is not None],
            key=lambda p: (float(p.x), float(p.y))
        )

    # Sort the synthetic facility DataFrame if present
    if 'dwater_gdf' in globals() and dwater_gdf is not None and not dwater_gdf.empty:
        def _xy(g):
            try:
                return (float(g.x), float(g.y))
            except Exception:
                c = g.centroid
                return (float(c.x), float(c.y))
        dwater_gdf = (
            dwater_gdf
            .assign(_xy=dwater_gdf.geometry.apply(_xy))
            .sort_values('_xy')
            .drop(columns=['_xy'])
            .reset_index(drop=True)
        )

    # Sort each POI pool in POI_POOLS for reproducibility
    if 'POI_POOLS' in globals() and isinstance(POI_POOLS, dict):
        for k, v in POI_POOLS.items():
            if not v:
                continue
            POI_POOLS[k] = sorted(
                [pt for pt in v if pt is not None],
                key=lambda p: (float(p.x), float(p.y))
            )
    if 'MOBILITY_POINTS' in globals() and MOBILITY_POINTS:
        _mw = np.asarray(MOBILITY_WEIGHTS, dtype=float) if ('MOBILITY_WEIGHTS' in globals() and MOBILITY_WEIGHTS is not None) else np.ones(len(MOBILITY_POINTS), dtype=float)
        _pairs = [(pt, w) for pt, w in zip(MOBILITY_POINTS, _mw) if pt is not None]
        _pairs.sort(key=lambda pw: (float(pw[0].x), float(pw[0].y)))
        MOBILITY_POINTS = [pw[0] for pw in _pairs]
        MOBILITY_WEIGHTS = np.asarray([pw[1] for pw in _pairs], dtype=float) if _pairs else None

# Call once now (safe if some objects aren’t defined yet)
freeze_iterable_orders()

# Canonical scenario labels used by all diagnostics/plots.
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "intervB_mobilityreduce": "Intervention B\n(current)",
    "intervD_destinations_only": "Intervention D1\nactivity only",
    "intervD_facilities_only": "Intervention D2\ntaps + toilets",
    "intervD_area_redistribute": "Intervention D3\nall amenities",
}
SCENARIO_ORDER = list(SCENARIO_LABELS.keys())

def _scenario_tag_from_intervention(intervention):
    if not intervention:
        return "baseline"
    if isinstance(intervention, dict) and intervention.get("scenario_tag"):
        return str(intervention.get("scenario_tag"))
    typ = intervention.get("type") if isinstance(intervention, dict) else None
    if typ == "mobility_reduce" and intervention.get("max_radius_m", None) == 1500:
        return "interv3_radiuscap1500"
    if typ == "mobility_reduce":
        return "intervB_mobilityreduce"
    if typ == "area_redistribute_plus_latrines":
        return "interv4_area_redistribute_plus_latrines"
    if typ == "area_redistribute":
        mode = _intervention_d_mode_name(intervention) if '_intervention_d_mode_name' in globals() else str(intervention.get("decentralisation_mode", "all"))
        if mode == "destinations_only":
            return "intervD_destinations_only"
        if mode == "sanitation_only":
            return "intervD_facilities_only"
        return "intervD_area_redistribute"
    return str(typ or "baseline")

def _distance_bin(dist_m):
    try:
        d = float(dist_m)
    except Exception:
        return "unknown"
    if d <= 0:
        return "0"
    if d < 250:
        return "0-250m"
    if d < 1000:
        return "250m-1km"
    if d < 3000:
        return "1-3km"
    if d < 5000:
        return "3-5km"
    return ">5km"

def _classify_destination_type(pt):
    if pt is None:
        return "unknown"
    try:
        w = getattr(pt, 'wkb', None)
        if w in set(getattr(p, 'wkb', None) for p in globals().get('tap_points', []) if p is not None):
            return "tap"
        if w in set(getattr(p, 'wkb', None) for p in globals().get('toilet_geoms', []) if p is not None):
            return "toilet"
        if w in set(getattr(p, 'wkb', None) for p in globals().get('school_points', []) if p is not None):
            return "school"
        if w in set(getattr(p, 'wkb', None) for p in globals().get('market_points', []) if p is not None):
            return "market"
        return "generic"
    except Exception:
        return "unknown"

def _scenario_label(scenario):
    return SCENARIO_LABELS.get(str(scenario), str(scenario))

def _canonical_scenario_order(existing=None):
    if existing is None:
        return list(SCENARIO_ORDER)
    existing = set(existing)
    ordered = [k for k in SCENARIO_ORDER if k in existing]
    ordered.extend(sorted(existing - set(ordered)))
    return ordered

# ===================== Notebook cell 39 =====================
# R1 — CITY CENTER / BOUNDS  (density-derived boundary first)
from shapely.ops import unary_union as _unary_union
import geopandas as gpd

def _union_geom(g):
    """Prefer GeoPandas union_all(); fallback to shapely.ops.unary_union."""
    if isinstance(g, gpd.GeoDataFrame) or isinstance(g, gpd.GeoSeries):
        if hasattr(g, "union_all"):
            return g.union_all()
        geoms = g.geometry if hasattr(g, "geometry") else g
        return _unary_union(geoms)
    return g  # if already a shapely geometry

# --- choose base boundary: density-derived → 7 km → old fallbacks ---
_base = None

if "boundaries_by_radius" in globals():
    if "R_GLOBAL_M" in globals() and int(R_GLOBAL_M) in boundaries_by_radius:
        _base = boundaries_by_radius[int(R_GLOBAL_M)]
    elif 7_000 in boundaries_by_radius:
        _base = boundaries_by_radius[7_000]

# older fallbacks if needed
if _base is None:
    if 'gdf_bound_3857' in globals() and gdf_bound_3857 is not None and not gdf_bound_3857.empty:
        _base = gdf_bound_3857
    elif 'inner_gdf' in globals() and inner_gdf is not None and not inner_gdf.empty:
        _base = inner_gdf

# harmonize CRS to EPSG:3857
if _base is not None and hasattr(_base, "crs") and _base.crs is not None and _base.crs.to_string() != "EPSG:3857":
    _base = _base.to_crs(3857)

# derive city polygon, centroid, bounds
if _base is not None and not _base.empty:
    CITY_POLY   = _union_geom(_base)
    center_pt   = CITY_POLY.centroid
    city_bounds = _base.total_bounds
else:
    CITY_POLY   = None
    center_pt   = None
    city_bounds = None

# ===================== Notebook cell 40 =====================
# R2a — Select the 1,000-agent analysis cohort
import random
if 'full_agent_list' in globals():
    desired_n = 1000
    agent_list = random.sample(full_agent_list, min(desired_n, len(full_agent_list)))
    print(f"Using downsampled cohort from full_agent_list: {len(agent_list)} agents")
elif 'agent_list' in globals():
    desired_n = 1000
    if len(agent_list) > desired_n:
        agent_list = random.sample(agent_list, desired_n)
    print(f"Using capped agent_list: {len(agent_list)} agents")
else:
    raise RuntimeError("No agent roster found. Run your agent-building cells first.")

# ===================== Notebook cell 41 =====================
# R2 — RESET STATE (agents + contamination) — with robust TAP + WATER seeding
import math

def reset_agents_and_contamination(seed=RUN_SEED):
    set_seed(seed)

    # Ensure deterministic ordering of taps/POIs if they exist
    if 'freeze_iterable_orders' in globals():
        freeze_iterable_orders()

    # --- Agents: reset states, locations, timers and ALL mobility state ---
    # Every scenario must start independently from the same baseline mobility
    # parameters and empty destination memory. Otherwise a scenario can inherit
    # exploration parameters or visited places from the scenario run before it.
    for ag in agent_list:
        ag.state = "S"; ag.days_in_state = 0
        ag.current_location = ag.home_location
        if hasattr(ag, "hydration_level"): ag.hydration_level = 100.0
        if hasattr(ag, "feces_timer"):     ag.feces_timer = max(1, int(np.random.normal(8, 2)))
        ag.exploration_prob = getattr(ag, "_rho_baseline", globals().get("BASE_RHO_MEAN", 0.5))
        ag.exploration_decay = getattr(ag, "_gamma_baseline", globals().get("BASE_GAMMA_MEAN", 0.7))
        ag.visited_places = []
        ag.destination = None
        # Journey metadata used by the co-occurrence visit diary.
        ag.destination_type = None
        ag.destination_id = None
        ag.departure_tick = None
        ag.departure_location = None

    # Seeding: scenario-controlled initial Exposed/Infectious fractions.
    # Keep initial seeding modest so downstream transmission drives the curves.
    n = len(agent_list)
    nE = max(1, int(float(globals().get("INITIAL_EXPOSED_FRAC", 0.01)) * n))
    nI = max(1, int(float(globals().get("INITIAL_INFECTED_FRAC", 0.02)) * n))
    pool = agent_list.copy()
    exposed = random.sample(pool, nE)
    pool = [a for a in pool if a not in exposed]
    infected = random.sample(pool, nI)
    for a in exposed:  a.state, a.days_in_state = "E", 0
    for a in infected: a.state, a.days_in_state = "I", 0

    # --- Build contamination registries fresh ---
    global tap_contamination, tap_contamination_by_wkb
    global water_contamination, water_contamination_by_wkb
    global home_contamination, contaminated_homes

    tap_contamination = {}  # tap-only epidemiological registry
    threshold_per_m2_local = globals().get("threshold_per_m2", 1e9)

    def _put_tap_rec(geom, feces_load=0.0, contaminated=False, area=1.0, threshold_total=None):
        if threshold_total is None:
            threshold_total = threshold_per_m2_local * max(float(area), 1.0)
        tap_contamination[id(geom)] = dict(
            geometry=geom,
            feces_load=float(feces_load),
            contaminated=bool(contaminated),
            area=float(area),
            threshold_total=float(threshold_total),
            # These fields record the provenance of the
            # existing binary contamination state without changing its logic.
            initially_contaminated=bool(contaminated),
            dynamically_contaminated=False,
            contamination_tick=None,
        )

    # Drinking water taps: unify sources (dwater_gdf and/or drinking_points)
    tap_geoms = []
    if 'dwater_gdf' in globals() and dwater_gdf is not None and not dwater_gdf.empty:
        tap_geoms.extend(list(dwater_gdf.geometry))
    if 'drinking_points' in globals() and drinking_points:
        tap_geoms.extend([pt for pt in drinking_points if pt is not None])

    _seen_wkb = set()
    _tap_unique = []
    for pt in tap_geoms:
        try:
            w = pt.wkb
        except Exception:
            continue
        if w not in _seen_wkb:
            _seen_wkb.add(w)
            _tap_unique.append(pt)

    _tap_unique = sorted(_tap_unique, key=lambda pt: (float(pt.x), float(pt.y)))

    n_taps = len(_tap_unique)
    _baseline_tap_wkbs = set(globals().get('BASELINE_TAP_WKBS', []) or [])
    _baseline_candidates = [pt for pt in _tap_unique if getattr(pt, 'wkb', None) in _baseline_tap_wkbs] if _baseline_tap_wkbs else list(_tap_unique)
    n_hot  = max(1, math.ceil(float(globals().get("TAP_INITIAL_CONTAMINATION_FRAC", 0.10)) * len(_baseline_candidates))) if _baseline_candidates else 0
    _hot_wkbs = set(getattr(pt, 'wkb', None) for pt in random.sample(_baseline_candidates, n_hot)) if n_hot else set()

    for i, pt in enumerate(_tap_unique):
        is_hot = (getattr(pt, 'wkb', None) in _hot_wkbs)
        _put_tap_rec(
            pt,
            feces_load=(1e10 if is_hot else 0.0),
            contaminated=is_hot,
            area=1.0,
            threshold_total=threshold_per_m2_local * 1.0,
        )

    tap_contamination_by_wkb = {rec["geometry"].wkb: rec for rec in tap_contamination.values()}

    # Backward-compatible aliases used elsewhere in plotting/export code.
    water_contamination = tap_contamination
    water_contamination_by_wkb = tap_contamination_by_wkb

    # Home contamination registry (keep existing pathway, but separate from taps)
    household_ids = sorted({int(getattr(a, "household_id")) for a in agent_list if hasattr(a, "household_id")})
    n_home_hot = max(1, int(float(globals().get("HOME_INITIAL_CONTAMINATION_FRAC", 0.012)) * len(household_ids))) if household_ids else 0
    contaminated_homes = set(random.sample(household_ids, n_home_hot)) if n_home_hot else set()
    home_contamination = {
        hid: {"contaminated": (hid in contaminated_homes)} for hid in household_ids
    }

    print(f"[RESET] Agents: N={n} (E={nE}, I={nI})")
    print("[RESET] Natural water polygons registered epidemiologically: 0")
    print(f"[RESET] Taps registered: {n_taps} (contaminated at t=0: {n_hot} = {100.0*n_hot/max(1,n_taps):.1f}%)")
    print(f"[RESET] Homes registered: {len(household_ids)} (contaminated at t=0: {n_home_hot})")

# ── (CHANGED) Call the reset so it actually runs and prints ─────────
# Ensure R2a (cohort selection) has created `agent_list` before calling this.
assert 'agent_list' in globals() and len(agent_list) > 0, "Run R2a to create `agent_list` first."
reset_agents_and_contamination(seed=RUN_SEED)

# ===================== Notebook cell 42 =====================
# R3 — SCENARIO RUNNER (ticks-based) — logs SE & EI transitions, OD flows, daily trips

from collections import defaultdict, Counter
import pandas as pd, geopandas as gpd
from tqdm.auto import tqdm
import numpy as np
import random


def _normalised_share_vector(values, fallback_len=None):
    """Return a safe non-negative share vector that sums to 1."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 and fallback_len is not None:
        arr = np.ones(int(fallback_len), dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0.0:
        n = int(arr.size if arr.size else (fallback_len or 0))
        return np.full(n, 1.0 / max(n, 1), dtype=float)
    return arr / s


def _equal_hub_share(n_hubs):
    n = int(max(0, n_hubs))
    return np.full(n, 1.0 / max(n, 1), dtype=float)


def _blend_hub_shares(base_share, target_share, strength):
    """Blend a baseline hub share vector toward a target/equalised vector."""
    base = _normalised_share_vector(base_share)
    target = _normalised_share_vector(target_share, fallback_len=len(base))
    if len(target) != len(base):
        target = _equal_hub_share(len(base))
    try:
        s = float(strength)
    except Exception:
        s = 0.0
    s = float(np.clip(s, 0.0, 1.0))
    out = (1.0 - s) * base + s * target
    return _normalised_share_vector(out, fallback_len=len(base))


def _d_area_equalisation_strength(k_geom=1.0):
    """Map k_geom onto a bounded 0..1 hub-area equalisation strength.

    k_geom is retained for backward compatibility with old grids/Sobol, but it no
    longer means 'shrink the dominant hub and boost a few targets'. Instead:
      0.0   -> keep baseline hub areas
      ~2.5+ -> move strongly toward equal hub inner-area shares
    """
    try:
        k = float(k_geom)
    except Exception:
        k = 1.0
    if not np.isfinite(k):
        k = 1.0
    return float(np.clip(k / 2.5, 0.0, 1.0))


def _intervention_d_mode_name(intervention_or_mode=None):
    """Canonicalise the three D variants.

    destinations_only = move schools, markets and generic activity POIs only.
    sanitation_only   = move drinking-water taps and toilets only.
    all               = move both activity destinations and WASH facilities.
    """
    if isinstance(intervention_or_mode, dict):
        raw = intervention_or_mode.get("decentralisation_mode", intervention_or_mode.get("d_mode", "all"))
    else:
        raw = intervention_or_mode if intervention_or_mode is not None else "all"
    raw = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "destinations": "destinations_only",
        "destination": "destinations_only",
        "activity": "destinations_only",
        "activity_only": "destinations_only",
        "schools_markets_pois": "destinations_only",
        "pois_only": "destinations_only",
        "poi_only": "destinations_only",
        "destinations_only": "destinations_only",
        "facilities_only": "sanitation_only",
        "facility_only": "sanitation_only",
        "wash_only": "sanitation_only",
        "sanitation_only": "sanitation_only",
        "taps_toilets": "sanitation_only",
        "taps_toilets_only": "sanitation_only",
        "water_sanitation_only": "sanitation_only",
        "everything": "all",
        "move_everything": "all",
        "all": "all",
    }
    return aliases.get(raw, "all")


def _select_area_redistribution_targets(inner_base, k_geom=1.0):
    """
    Equalised Intervention D geometry diagnostics.

    Kept under the old function name so downstream code does not break, but the
    logic is now final-paper aligned: all non-dominant hubs are valid secondary
    hubs and geometry moves toward more even hub inner-area shares. It no longer
    chooses a fixed small set of targets or transfers dominance to one secondary
    hub.
    """
    if inner_base is None or getattr(inner_base, "empty", True):
        return None

    work = inner_base.copy().reset_index(drop=True)
    work["hub_area_m2"] = work.geometry.area.astype(float)
    work = work.loc[np.isfinite(work["hub_area_m2"])].copy()
    if work.empty:
        return None

    dominant_idx = int(work["hub_area_m2"].idxmax())
    A = work["hub_area_m2"].to_numpy(dtype=float)
    A1 = float(A[dominant_idx]) if A.size else np.nan
    others = np.delete(A, dominant_idx) if A.size > 1 else np.asarray([], dtype=float)
    A2 = float(np.nanmax(others)) if others.size else np.nan
    Amed = float(np.nanmedian(others)) if others.size else np.nan

    strength = _d_area_equalisation_strength(k_geom)
    base_share = _normalised_share_vector(A, fallback_len=len(work))
    equal_share = _equal_hub_share(len(work))
    target_share = _blend_hub_shares(base_share, equal_share, strength)
    total_area = float(np.nansum(A)) if np.isfinite(np.nansum(A)) else 0.0
    target_area = target_share * max(total_area, 1.0)

    target_indices = [int(i) for i in range(len(work)) if int(i) != dominant_idx]
    delta_A = float(max(0.0, A1 - target_area[dominant_idx])) if np.isfinite(A1) and len(target_area) else 0.0
    target_ratio = float(target_area[dominant_idx] / max(np.nanmax(np.delete(target_area, dominant_idx)), 1.0)) if len(target_area) > 1 else np.nan

    work["target_area_m2"] = target_area
    work["baseline_area_share"] = base_share
    work["equal_area_share"] = equal_share
    work["target_area_share"] = target_share

    return {
        "work": work,
        "dominant_idx": dominant_idx,
        "target_indices": target_indices,
        "shrink_frac": float(delta_A / max(A1, 1.0)) if np.isfinite(A1) else 0.0,
        "shrink_frac_base": float(strength),
        "k_geom": float(k_geom),
        "area_equalisation_strength": float(strength),
        "A1": A1,
        "A2": A2,
        "Amed": Amed,
        "A1_floor": np.nan,
        "A1_new": float(target_area[dominant_idx]) if len(target_area) else A1,
        "delta_A": delta_A,
        "target_ratio": target_ratio,
        "n_targets": int(len(target_indices)),
        "use_secondary_score_ranking": False,
        "secondary_priority_strength": np.nan,
        "n_possible_secondary_targets": int(max(0, len(work) - 1)),
        "baseline_area_share": base_share,
        "target_area_share": target_share,
    }

def _is_geometry_object(obj):
    return (obj is not None) and hasattr(obj, "geom_type") and hasattr(obj, "is_empty")


def _clean_geometry_object(obj, fallback=None):
    cand = obj if _is_geometry_object(obj) else fallback
    if not _is_geometry_object(cand):
        return None
    try:
        if cand.is_empty:
            return None
    except Exception:
        return None

    # IMPORTANT: do not call buffer(0) on Points or other non-polygonal
    # geometries. In Shapely, Point.buffer(0) returns an empty polygon, which
    # silently deletes tap/toilet/school/market/destination points from the
    # intervention payloads. Only polygonal geometries need this repair.
    try:
        if getattr(cand, 'geom_type', None) in ('Polygon', 'MultiPolygon'):
            repaired = cand.buffer(0)
            if _is_geometry_object(repaired) and not repaired.is_empty:
                cand = repaired
    except Exception:
        pass

    return cand if _is_geometry_object(cand) and not cand.is_empty else None


def _clean_geometry_list(seq):
    out = []
    for obj in (seq or []):
        g = _clean_geometry_object(obj)
        if g is not None:
            out.append(g)
    return out


def _strip_holes_polygonal(geom):
    geom = _clean_geometry_object(geom)
    if geom is None:
        return None
    try:
        if geom.geom_type == "Polygon":
            return Polygon(geom.exterior)
        if geom.geom_type == "MultiPolygon":
            parts = []
            for g in geom.geoms:
                gg = _clean_geometry_object(g)
                if gg is not None and gg.geom_type == "Polygon":
                    parts.append(Polygon(gg.exterior))
            if not parts:
                return geom
            try:
                return MultiPolygon(parts)
            except Exception:
                return unary_union(parts)
    except Exception:
        return geom
    return geom


def _empty_gdf_like(crs=None, extra_cols=None):
    extra_cols = list(extra_cols or [])
    data = {c: pd.Series(dtype="float64") for c in extra_cols}
    data["geometry"] = gpd.GeoSeries([], crs=crs)
    return gpd.GeoDataFrame(data, geometry="geometry", crs=crs)


def _build_cell_boundaries_from_inner_outer(inner_src, outer_src):
    """Return full hub-cell polygons so plotting outer guides does not re-draw old inner-hole boundaries.
    Uses cluster matching when available; otherwise falls back to representative-point matching.
    """
    if inner_src is None and outer_src is None:
        return None
    try:
        import geopandas as _gpd
    except Exception:
        return outer_src

    if outer_src is None or getattr(outer_src, 'empty', True):
        return outer_src
    if inner_src is None or getattr(inner_src, 'empty', True):
        return outer_src

    inner_df = inner_src.copy()
    outer_df = outer_src.copy()
    target_crs = getattr(outer_df, 'crs', None) or getattr(inner_df, 'crs', None)
    try:
        if target_crs is not None and getattr(inner_df, 'crs', None) is not None and inner_df.crs != target_crs:
            inner_df = inner_df.to_crs(target_crs)
    except Exception:
        pass
    try:
        if target_crs is not None and getattr(outer_df, 'crs', None) is not None and outer_df.crs != target_crs:
            outer_df = outer_df.to_crs(target_crs)
    except Exception:
        pass

    rows = []
    used_inner = set()
    inner_by_cluster = {}
    if 'cluster' in outer_df.columns and 'cluster' in inner_df.columns:
        try:
            for ii, r in inner_df.iterrows():
                if pd.notna(r.get('cluster', None)):
                    inner_by_cluster[int(r['cluster'])] = ii
        except Exception:
            inner_by_cluster = {}

    for _, orow in outer_df.iterrows():
        outer_geom = _clean_geometry_object(orow.geometry)
        if outer_geom is None:
            continue
        match_geom = None
        cid = None
        if 'cluster' in orow.index and pd.notna(orow.get('cluster', None)):
            try:
                cid = int(orow['cluster'])
                if cid in inner_by_cluster:
                    ii = inner_by_cluster[cid]
                    match_geom = _clean_geometry_object(inner_df.loc[ii].geometry)
                    used_inner.add(ii)
            except Exception:
                match_geom = None
        if match_geom is None:
            rep = None
            try:
                rep = outer_geom.representative_point()
            except Exception:
                rep = None
            if rep is not None:
                for ii, irow in inner_df.iterrows():
                    if ii in used_inner:
                        continue
                    ig = _clean_geometry_object(irow.geometry)
                    if ig is None:
                        continue
                    try:
                        if outer_geom.distance(ig) <= 1e-6 or outer_geom.buffer(1e-6).contains(rep) or ig.distance(outer_geom) <= 1e-6:
                            # prefer same cell by nearest representative point
                            pass
                    except Exception:
                        pass
                try:
                    d = inner_df.geometry.distance(rep)
                    if len(d):
                        ii = int(d.idxmin())
                        match_geom = _clean_geometry_object(inner_df.loc[ii].geometry)
                        used_inner.add(ii)
                except Exception:
                    match_geom = None
        cell_geom = outer_geom
        if match_geom is not None:
            try:
                cell_geom = _clean_geometry_object(outer_geom.union(match_geom), fallback=outer_geom)
            except Exception:
                cell_geom = outer_geom
        rowd = {}
        if 'cluster' in orow.index:
            rowd['cluster'] = orow.get('cluster', cid)
        rowd['geometry'] = _strip_holes_polygonal(cell_geom) or cell_geom
        rows.append(rowd)

    if not rows:
        return outer_df
    return _gpd.GeoDataFrame(rows, geometry='geometry', crs=target_crs)


def _build_area_redistributed_inner_gdf(inner_base, outer_base, seed=0, k_geom=1.0):
    """
    Final-paper Intervention D geometry helper.

    This now changes hub *sizes* by equalising inner-hub area shares across all
    existing hubs, rather than shrinking the main hub and expanding only a few
    selected secondary hubs. The total inner-hub area is kept approximately fixed;
    large hubs shrink and smaller secondary hubs grow inside their own Voronoi
    catchments.
    """
    if inner_base is None or getattr(inner_base, "empty", True):
        return inner_base
    if outer_base is None or getattr(outer_base, "empty", True):
        return inner_base.copy()

    try:
        import geopandas as _gpd
    except Exception:
        return inner_base.copy()

    inner_work = inner_base.copy()
    outer_work = outer_base.copy()
    original_crs = getattr(inner_work, "crs", None)
    if getattr(inner_work, "crs", None) is not None and str(inner_work.crs) != "EPSG:3857":
        inner_work = inner_work.to_crs("EPSG:3857")
    if getattr(outer_work, "crs", None) is not None and str(outer_work.crs) != "EPSG:3857":
        outer_work = outer_work.to_crs("EPSG:3857")

    inner_work = inner_work.reset_index(drop=True)
    outer_work = outer_work.reset_index(drop=True)
    if inner_work.empty:
        return inner_base.copy()

    # Full catchment/cell polygon for each hub = outer part + old inner part.
    cell_gdf = _build_cell_boundaries_from_inner_outer(inner_work, outer_work)
    if cell_gdf is None or getattr(cell_gdf, "empty", True):
        cell_gdf = outer_work.copy()
    if getattr(cell_gdf, "crs", None) is not None and str(cell_gdf.crs) != "EPSG:3857":
        cell_gdf = cell_gdf.to_crs("EPSG:3857")
    cell_gdf = cell_gdf.reset_index(drop=True)

    # Match cells to inner rows by cluster where possible; otherwise row order.
    cell_by_cluster = {}
    if "cluster" in cell_gdf.columns:
        for ii, rr in cell_gdf.iterrows():
            try:
                cell_by_cluster[int(rr["cluster"])] = rr.geometry
            except Exception:
                pass

    cells = []
    for i, row in inner_work.iterrows():
        cid = int(row["cluster"]) if "cluster" in inner_work.columns and pd.notna(row.get("cluster", np.nan)) else int(i)
        cell = cell_by_cluster.get(cid, None)
        if cell is None and i < len(cell_gdf):
            cell = cell_gdf.geometry.iloc[int(i)]
        if cell is None:
            cell = row.geometry
        cell = _clean_geometry_object(cell, fallback=row.geometry)
        cells.append(cell if cell is not None else row.geometry)

    current_area = inner_work.geometry.area.astype(float).to_numpy(dtype=float)
    current_area[~np.isfinite(current_area)] = 0.0
    total_inner_area = float(np.sum(np.clip(current_area, 0.0, None)))
    if total_inner_area <= 0:
        return inner_base.copy()

    strength = _d_area_equalisation_strength(k_geom)
    base_share = _normalised_share_vector(current_area, fallback_len=len(inner_work))
    target_share = _blend_hub_shares(base_share, _equal_hub_share(len(inner_work)), strength)
    target_area = target_share * total_inner_area

    # Respect cell capacity so a secondary hub cannot grow outside its own catchment.
    cell_area = np.asarray([float(g.area) if _is_geometry_object(g) and not g.is_empty else 0.0 for g in cells], dtype=float)
    cap = np.maximum(1.0, 0.96 * cell_area)
    target_area = np.minimum(target_area, cap)
    # If caps reduced the total, redistribute remaining to hubs with spare cell capacity.
    for _ in range(4):
        deficit = total_inner_area - float(target_area.sum())
        if deficit <= 1e-6:
            break
        spare = np.maximum(cap - target_area, 0.0)
        if float(spare.sum()) <= 1e-6:
            break
        target_area += deficit * (spare / float(spare.sum()))
        target_area = np.minimum(target_area, cap)

    def _grow_to_target(cell_poly, seed_pt, target_a):
        cell_poly = _clean_geometry_object(cell_poly)
        if cell_poly is None:
            return None
        max_area = float(max(cell_poly.area, 1.0))
        tgt = float(np.clip(float(target_a), 1.0, 0.995 * max_area))
        if tgt >= 0.990 * max_area:
            return cell_poly
        if seed_pt is None or not _is_geometry_object(seed_pt):
            seed_pt = cell_poly.representative_point()
        # Radius search around the old hub representative point. This changes size
        # but keeps the hub inside its own cell.
        lo = 0.0
        hi = max(1.0, np.sqrt(max_area / np.pi))
        for _ in range(50):
            cand = _clean_geometry_object(cell_poly.intersection(seed_pt.buffer(hi)))
            if cand is not None and cand.area >= tgt:
                break
            hi *= 1.5
            if hi > 1e7:
                break
        best = None
        for _ in range(55):
            mid = 0.5 * (lo + hi)
            cand = _clean_geometry_object(cell_poly.intersection(seed_pt.buffer(mid)))
            a = float(cand.area) if cand is not None else 0.0
            if a >= tgt:
                best = cand
                hi = mid
            else:
                lo = mid
        if best is None:
            best = _clean_geometry_object(cell_poly.intersection(seed_pt.buffer(hi)), fallback=cell_poly)
        return best if best is not None and not best.is_empty else cell_poly

    new_rows = []
    for i, row in inner_work.iterrows():
        row_geom = _clean_geometry_object(row.geometry)
        if row_geom is None:
            continue
        try:
            seed_pt = row_geom.representative_point()
        except Exception:
            seed_pt = row_geom.centroid if hasattr(row_geom, "centroid") else None
        new_geom = _grow_to_target(cells[int(i)], seed_pt, float(target_area[int(i)]))
        if new_geom is None:
            new_geom = row_geom
        rowd = row.to_dict()
        rowd["geometry"] = new_geom
        rowd["baseline_area_m2"] = float(current_area[int(i)])
        rowd["target_area_m2"] = float(target_area[int(i)])
        rowd["area_equalisation_strength"] = float(strength)
        new_rows.append(rowd)

    if not new_rows:
        return inner_base.copy()
    inner_new = _gpd.GeoDataFrame(new_rows, geometry="geometry", crs="EPSG:3857")
    inner_new = inner_new.loc[inner_new.geometry.apply(lambda g: _is_geometry_object(g) and (not g.is_empty))].copy()
    inner_new = inner_new.reset_index(drop=True)
    if original_crs is not None and str(original_crs) != "EPSG:3857":
        inner_new = inner_new.to_crs(original_crs)
    return inner_new


def _build_depr_pool_for_area_intervention(inner_src, attractiveness_mode="centrality", eta=1.0, beta=0.5, lambda_blend=0.55, delta_dest=0.0, decentralisation_mode="all"):
    """
    Build D-only generic destination weights while preserving the baseline destination universe.

    Final-paper logic: D equalises hub-level destination mass across *all* hubs.
    It no longer removes mass from the dominant hub and boosts a small selected
    set of secondary hubs. If decentralisation_mode='sanitation_only', generic
    destination weights are returned unchanged.
    """
    if inner_src is None or getattr(inner_src, "empty", True):
        return [], None

    mode = _intervention_d_mode_name(decentralisation_mode)

    base_points = []
    if 'depr_points' in globals() and depr_points is not None:
        try:
            base_points = _clean_geometry_list(list(depr_points))
        except Exception:
            base_points = []
    if not base_points:
        return [], None

    if 'depr_weights' in globals() and depr_weights is not None:
        try:
            base_weights = np.asarray(depr_weights, dtype=float).copy()
        except Exception:
            base_weights = np.ones(len(base_points), dtype=float)
    else:
        base_weights = np.ones(len(base_points), dtype=float)
    if base_weights.size != len(base_points):
        base_weights = np.ones(len(base_points), dtype=float)
    base_weights[~np.isfinite(base_weights)] = 0.0
    if float(base_weights.sum()) <= 0:
        base_weights = np.ones(len(base_points), dtype=float)
    base_weights = base_weights / float(base_weights.sum())

    if mode == "sanitation_only":
        return list(base_points), base_weights

    inner_df = inner_src.copy().reset_index(drop=True)
    if inner_df.empty:
        return list(base_points), base_weights

    try:
        _sidx = inner_df.sindex
    except Exception:
        _sidx = None
    rep_pts = [geom.representative_point() if _is_geometry_object(geom) else None for geom in inner_df.geometry]

    point_hub_idx = []
    for pt in base_points:
        idx_hit = None
        if _sidx is not None:
            try:
                hits = list(_sidx.query(pt, predicate='contains'))
            except Exception:
                hits = []
            if hits:
                idx_hit = int(hits[0])
        if idx_hit is None:
            for j, geom in enumerate(inner_df.geometry):
                try:
                    if geom.contains(pt):
                        idx_hit = int(j)
                        break
                except Exception:
                    continue
        if idx_hit is None:
            d_best = None
            for j, rp in enumerate(rep_pts):
                if rp is None:
                    continue
                try:
                    d = pt.distance(rp)
                except Exception:
                    continue
                if d_best is None or d < d_best[0]:
                    d_best = (d, j)
            idx_hit = int(d_best[1]) if d_best is not None else 0
        point_hub_idx.append(idx_hit)

    point_hub_idx = np.asarray(point_hub_idx, dtype=int)
    n_hubs = len(inner_df)

    # Keep within-hub centrality preference, but equalise the hub-level mass.
    base_norm = base_weights / max(float(base_weights.max()), 1e-12)
    point_pref = np.power(np.maximum(base_norm, 1e-9), max(float(beta), 1.0))
    point_pref = np.asarray(point_pref, dtype=float) * np.maximum(base_weights, 1e-12)
    point_pref[~np.isfinite(point_pref)] = 0.0
    if float(point_pref.sum()) <= 0:
        point_pref = np.asarray(base_weights, dtype=float)

    base_hub_mass = np.zeros(n_hubs, dtype=float)
    hub_centrality_mass = np.zeros(n_hubs, dtype=float)
    for j in range(n_hubs):
        mask = point_hub_idx == j
        if np.any(mask):
            base_hub_mass[j] = float(base_weights[mask].sum())
            hub_centrality_mass[j] = float(point_pref[mask].sum())
    base_hub_mass = _normalised_share_vector(base_hub_mass, fallback_len=n_hubs)
    hub_score = _normalised_share_vector(hub_centrality_mass, fallback_len=n_hubs)

    lambda_blend = float(np.clip(lambda_blend, 0.0, 1.0))
    pre_intervention_hub_mass = _normalised_share_vector(
        lambda_blend * base_hub_mass + (1.0 - lambda_blend) * hub_score,
        fallback_len=n_hubs,
    )

    try:
        dd = float(delta_dest)
    except Exception:
        dd = 0.0
    dd = float(np.clip(dd, 0.0, 1.0))
    hub_mass = _blend_hub_shares(pre_intervention_hub_mass, _equal_hub_share(n_hubs), dd)

    # Hubs with no baseline generic point cannot receive weight unless a point exists.
    # Reassign their small equalised mass back to hubs that have points, preserving
    # a valid probability vector over the existing point universe.
    has_points = np.asarray([np.any(point_hub_idx == j) for j in range(n_hubs)], dtype=bool)
    if np.any(has_points) and not np.all(has_points):
        hub_mass = hub_mass * has_points.astype(float)
        hub_mass = _normalised_share_vector(hub_mass, fallback_len=n_hubs)

    score = np.zeros(len(base_points), dtype=float)
    for j in range(n_hubs):
        mask = point_hub_idx == j
        if not np.any(mask):
            continue
        within = point_pref[mask].copy()
        within[~np.isfinite(within)] = 0.0
        if float(within.sum()) <= 0:
            within = np.ones(mask.sum(), dtype=float)
        within = within / float(within.sum())
        score[mask] = hub_mass[j] * within

    score[~np.isfinite(score)] = 0.0
    if float(score.sum()) <= 0:
        return list(base_points), base_weights
    score = score / float(score.sum())
    return list(base_points), score

def _allocate_counts_by_weights(total_count, weights, min_per=0, only_positive=True):
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0 or int(total_count) <= 0:
        return np.zeros(weights.size, dtype=int)
    weights[~np.isfinite(weights)] = 0.0
    if only_positive:
        active = weights > 0
    else:
        active = np.ones(weights.size, dtype=bool)
    if not np.any(active):
        active = np.ones(weights.size, dtype=bool)
        weights = np.ones(weights.size, dtype=float)
    base = np.zeros(weights.size, dtype=int)
    min_per = int(max(0, min_per))
    if min_per > 0:
        active_idx = np.where(active)[0]
        if len(active_idx) * min_per <= int(total_count):
            base[active_idx] = min_per
        else:
            min_per = 0
    remaining = int(total_count) - int(base.sum())
    if remaining <= 0:
        return base
    w = np.clip(weights.astype(float), 0.0, None)
    if float(w.sum()) <= 0:
        w = np.ones_like(w, dtype=float)
    w = w / w.sum()
    raw = remaining * w
    extra = np.floor(raw).astype(int)
    out = base + extra
    left = int(total_count) - int(out.sum())
    if left > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for j in order[:left]:
            out[int(j)] += 1
    return out.astype(int)


def _random_points_in_polygon_intervention(poly, n, rng=None, max_tries=80):
    if n <= 0 or poly is None or poly.is_empty:
        return []
    import random as _random
    rr = rng if rng is not None else _random.Random(0)
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    tries = 0
    while len(pts) < int(n) and tries < int(n) * int(max_tries):
        pt = Point(rr.uniform(minx, maxx), rr.uniform(miny, maxy))
        try:
            inside = poly.contains(pt)
        except Exception:
            inside = False
        if inside:
            pts.append(pt)
        tries += 1
    while len(pts) < int(n):
        try:
            pts.append(poly.representative_point())
        except Exception:
            break
    return pts



def _build_area_redistributed_facilities(inner_src, seed=0, eta=1.90, beta=0.25, lambda_blend=0.55, alpha_fac=0.80, beta_fac=0.30, delta_fac=0.0, delta_dest=0.0, decentralisation_mode="all"):
    """
    D-only amenity redistribution with three variants:
      destinations_only: move schools, markets and generic activity POIs only.
      sanitation_only:   move taps and toilets only.
      all:               move activity destinations and WASH facilities.

    Citywide totals are kept fixed. Hub-level shares are blended toward equal
    shares across all hubs, rather than transferring dominance from one hub to a
    small selected set.
    """
    if inner_src is None or getattr(inner_src, "empty", True):
        return None
    inner_df = inner_src.copy().reset_index(drop=True)
    if inner_df.empty:
        return None
    target_crs = getattr(inner_df, 'crs', None)
    mode = _intervention_d_mode_name(decentralisation_mode)
    n_hubs = len(inner_df)
    equal_share = _equal_hub_share(n_hubs)

    try:
        dd = float(delta_dest)
    except Exception:
        dd = 0.0
    try:
        dfac = float(delta_fac)
    except Exception:
        dfac = 0.0
    dd = float(np.clip(dd, 0.0, 1.0))
    dfac = float(np.clip(dfac, 0.0, 1.0))

    rr = random.Random(int(seed) + 17031)

    def _total_points(obj):
        if obj is None:
            return 0
        try:
            return int(len(obj))
        except Exception:
            return 0

    def _cluster_vector_from_gdf(gdf_obj):
        vec = np.zeros(n_hubs, dtype=float)
        if gdf_obj is None or getattr(gdf_obj, 'empty', True):
            return vec
        if 'cluster' in gdf_obj.columns and 'cluster' in inner_df.columns:
            counts = gdf_obj['cluster'].value_counts()
            for i, row in inner_df.iterrows():
                try:
                    cid = int(row['cluster'])
                except Exception:
                    cid = int(i)
                vec[int(i)] = float(counts.get(cid, 0.0))
            return vec
        # Fallback: spatially assign points to current intervention hubs.
        try:
            sidx = inner_df.sindex
        except Exception:
            sidx = None
        for geom in list(gdf_obj.geometry):
            hit = None
            if sidx is not None:
                try:
                    hits = list(sidx.query(geom, predicate='contains'))
                except Exception:
                    hits = []
                if hits:
                    hit = int(hits[0])
            if hit is None:
                for j, poly in enumerate(inner_df.geometry):
                    try:
                        if poly.contains(geom):
                            hit = int(j)
                            break
                    except Exception:
                        continue
            if hit is not None and 0 <= hit < n_hubs:
                vec[hit] += 1.0
        return vec

    def _counts_for_kind(total, base_vec, strength, move_this_kind, min_if_enough=True):
        total = int(max(0, total))
        if total <= 0:
            return np.zeros(n_hubs, dtype=int)
        base_share = _normalised_share_vector(base_vec, fallback_len=n_hubs)
        if not move_this_kind:
            # Keep old allocation as closely as possible.
            return _allocate_counts_by_weights(total, base_share, min_per=0, only_positive=False)
        share = _blend_hub_shares(base_share, equal_share, strength)
        min_per = 1 if (min_if_enough and total >= n_hubs) else 0
        return _allocate_counts_by_weights(total, share, min_per=min_per, only_positive=False)

    totals = {
        'water': _total_points(dwater_gdf) if 'dwater_gdf' in globals() and dwater_gdf is not None else (_total_points(drinking_points) if 'drinking_points' in globals() else 0),
        'school': _total_points(schools_gdf) if 'schools_gdf' in globals() and schools_gdf is not None else 0,
        'market': _total_points(markets_gdf) if 'markets_gdf' in globals() and markets_gdf is not None else 0,
        'toilet': _total_points(toilets_gdf) if 'toilets_gdf' in globals() and toilets_gdf is not None else 0,
    }

    water_base = _cluster_vector_from_gdf(dwater_gdf if 'dwater_gdf' in globals() else None)
    school_base = _cluster_vector_from_gdf(schools_gdf if 'schools_gdf' in globals() else None)
    market_base = _cluster_vector_from_gdf(markets_gdf if 'markets_gdf' in globals() else None)
    toilet_base = _cluster_vector_from_gdf(toilets_gdf if 'toilets_gdf' in globals() else None)

    move_destinations = mode in ("destinations_only", "all")
    move_sanitation = mode in ("sanitation_only", "all")
    moved_kinds = {
        'generic': bool(move_destinations),
        'school': bool(move_destinations),
        'market': bool(move_destinations),
        'water': bool(move_sanitation),
        'toilet': bool(move_sanitation),
    }

    allocs = {
        'water': _counts_for_kind(totals['water'], water_base, dfac, move_sanitation, min_if_enough=True),
        'school': _counts_for_kind(totals['school'], school_base, dd, move_destinations, min_if_enough=False),
        'market': _counts_for_kind(totals['market'], market_base, dd, move_destinations, min_if_enough=True),
        'toilet': _counts_for_kind(totals['toilet'], toilet_base, dfac, move_sanitation, min_if_enough=True),
    }

    def _make_gdf(kind, feature_type, cluster_col='cluster'):
        rows = []
        for i, row in inner_df.iterrows():
            n = int(allocs[kind][i]) if i < len(allocs[kind]) else 0
            if n <= 0:
                continue
            poly = _clean_geometry_object(row.geometry)
            if poly is None:
                continue
            pts = _clean_geometry_list(_random_points_in_polygon_intervention(poly, n, rng=rr, max_tries=80))
            cid = int(row[cluster_col]) if cluster_col in inner_df.columns and pd.notna(row[cluster_col]) else int(i)
            for p in pts:
                rows.append({'cluster': cid, 'feature_type': feature_type, 'geometry': p})
        if not rows:
            return _empty_gdf_like(crs=target_crs, extra_cols=['cluster', 'feature_type'])
        out = gpd.GeoDataFrame(rows, geometry='geometry', crs=target_crs)
        out = out.loc[out.geometry.apply(lambda g: _is_geometry_object(g) and (not g.is_empty))].copy()
        return out if not out.empty else _empty_gdf_like(crs=target_crs, extra_cols=['cluster', 'feature_type'])

    # Activity destination facilities.
    if move_destinations:
        schools_new = _make_gdf('school', 'School')
        markets_new = _make_gdf('market', 'Market')
    else:
        schools_new = schools_gdf.copy() if ('schools_gdf' in globals() and schools_gdf is not None) else _empty_gdf_like(crs=target_crs, extra_cols=['cluster', 'feature_type'])
        markets_new = markets_gdf.copy() if ('markets_gdf' in globals() and markets_gdf is not None) else _empty_gdf_like(crs=target_crs, extra_cols=['cluster', 'feature_type'])

    # WASH facilities.
    if move_sanitation:
        dwater_new = _make_gdf('water', 'Drinking Water')
        toilets_new = _make_gdf('toilet', 'Public Toilet')
    else:
        dwater_new = dwater_gdf.copy() if ('dwater_gdf' in globals() and dwater_gdf is not None) else _empty_gdf_like(crs=target_crs, extra_cols=['cluster', 'feature_type'])
        toilets_new = toilets_gdf.copy() if ('toilets_gdf' in globals() and toilets_gdf is not None) else _empty_gdf_like(crs=target_crs, extra_cols=['cluster', 'feature_type'])

    drinking_new = _clean_geometry_list(list(dwater_new.geometry)) if dwater_new is not None and not dwater_new.empty else []
    school_new = _clean_geometry_list(list(schools_new.geometry)) if schools_new is not None and not schools_new.empty else []
    market_new = _clean_geometry_list(list(markets_new.geometry)) if markets_new is not None and not markets_new.empty else []
    toilet_new = _clean_geometry_list(list(toilets_new.geometry)) if toilets_new is not None and not toilets_new.empty else []

    generic_new, generic_weights = _build_depr_pool_for_area_intervention(
        inner_df,
        attractiveness_mode='centrality',
        eta=float(eta),
        beta=float(beta),
        lambda_blend=float(lambda_blend),
        delta_dest=float(delta_dest),
        decentralisation_mode=mode,
    )

    _nodes_for_weights = None
    try:
        _nodes_key = _mobility_nodes_source_key() if '_mobility_nodes_source_key' in globals() else None
        if _nodes_key is not None and 'centrality_by_radius' in globals() and _nodes_key in centrality_by_radius:
            _nodes_for_weights = centrality_by_radius[_nodes_key].copy()
            if getattr(_nodes_for_weights, 'crs', None) != target_crs:
                _nodes_for_weights = _nodes_for_weights.to_crs(target_crs)
    except Exception:
        _nodes_for_weights = None

    mobility_points_new, mobility_weights_new = _build_weighted_mobility_pool(
        list(generic_new) if generic_new else [],
        None if generic_weights is None else np.asarray(generic_weights, dtype=float),
        school_new,
        market_new,
        _nodes_for_weights,
    ) if '_build_weighted_mobility_pool' in globals() else (list(generic_new) if generic_new else [], None if generic_weights is None else np.asarray(generic_weights, dtype=float))

    poi_pools_new = {
        'generic': list(mobility_points_new) if mobility_points_new else _clean_geometry_list([geom.representative_point() for geom in inner_df.geometry if _is_geometry_object(geom)]),
        'school': school_new,
        'market': market_new,
        'taps': drinking_new,
        'toilets': toilet_new,
    }

    share_rows = []
    for i, row in inner_df.iterrows():
        cid = int(row['cluster']) if 'cluster' in inner_df.columns and pd.notna(row.get('cluster', np.nan)) else int(i)
        share_rows.append({
            'cluster': cid,
            'baseline_water_count': float(water_base[i]) if i < len(water_base) else 0.0,
            'baseline_school_count': float(school_base[i]) if i < len(school_base) else 0.0,
            'baseline_market_count': float(market_base[i]) if i < len(market_base) else 0.0,
            'baseline_toilet_count': float(toilet_base[i]) if i < len(toilet_base) else 0.0,
            'alloc_water': int(allocs['water'][i]) if i < len(allocs['water']) else 0,
            'alloc_school': int(allocs['school'][i]) if i < len(allocs['school']) else 0,
            'alloc_market': int(allocs['market'][i]) if i < len(allocs['market']) else 0,
            'alloc_toilet': int(allocs['toilet'][i]) if i < len(allocs['toilet']) else 0,
            'equal_hub_share': float(equal_share[i]) if i < len(equal_share) else np.nan,
            'mode': mode,
        })
    share_table = pd.DataFrame(share_rows)

    return {
        'dwater_gdf': dwater_new,
        'schools_gdf': schools_new,
        'markets_gdf': markets_new,
        'toilets_gdf': toilets_new,
        'drinking_points': drinking_new,
        'school_points': school_new,
        'market_points': market_new,
        'toilet_points': toilet_new,
        # Keep true generic high-centrality/activity points separate from the
        # combined movement pool.  The movement pool intentionally also contains
        # schools and markets, but co-occurrence diagnostics must not count those
        # facilities twice.
        'generic_points': list(generic_new) if generic_new else [],
        'generic_weights': None if generic_weights is None else np.asarray(generic_weights, dtype=float),
        'POI_POOLS': poi_pools_new,
        'depr_points': list(mobility_points_new) if mobility_points_new else [],
        'depr_weights': None if mobility_weights_new is None else np.asarray(mobility_weights_new, dtype=float),
        'importance': equal_share,
        'alloc_water': allocs['water'],
        'alloc_school': allocs['school'],
        'alloc_market': allocs['market'],
        'alloc_toilet': allocs['toilet'],
        'mode': mode,
        'moved_kinds': moved_kinds,
        'share_table': share_table,
    }


def _refresh_tap_registry_for_current_taps(seed=None, preserve_existing_flags=False):
    """Rebuild tap contamination mirrors after D moves taps.

    preserve_existing_flags=True keeps contamination status by WKB where possible;
    otherwise the current tap set is seeded with the same global contamination
    fraction. New moved taps are not silently left unregistered.
    """
    global tap_contamination, tap_contamination_by_wkb, water_contamination, water_contamination_by_wkb
    taps = _clean_geometry_list(list(globals().get('drinking_points', [])))
    if seed is not None:
        _state = random.getstate()
        random.seed(int(seed) + 7001)
    else:
        _state = None
    try:
        old_by_wkb = globals().get('tap_contamination_by_wkb', {}) if preserve_existing_flags else {}
        threshold_local = globals().get('threshold_per_m2', 1e9)
        tap_contamination = {}
        if preserve_existing_flags and old_by_wkb:
            hot_wkbs = {w for w, rec in old_by_wkb.items() if bool(rec.get('contaminated', False))}
        else:
            frac = float(globals().get("TAP_INITIAL_CONTAMINATION_FRAC", 0.10))
            n_hot = max(1, int(np.ceil(frac * len(taps)))) if taps else 0
            hot_wkbs = set(getattr(pt, 'wkb', None) for pt in random.sample(taps, min(n_hot, len(taps)))) if n_hot else set()
        for pt in taps:
            hot = getattr(pt, 'wkb', None) in hot_wkbs
            tap_contamination[id(pt)] = {
                'geometry': pt,
                'feces_load': 1e10 if hot else 0.0,
                'area': 1.0,
                'threshold_total': 1.0 if hot else threshold_local * 1.0,
                'contaminated': bool(hot),
                # Initial contamination is drawn for each replicate;
                # the following fields identify its origin.
                'initially_contaminated': bool(hot),
                'dynamically_contaminated': False,
                'contamination_tick': None,
            }
        tap_contamination_by_wkb = {rec['geometry'].wkb: rec for rec in tap_contamination.values()}
        water_contamination = tap_contamination
        water_contamination_by_wkb = tap_contamination_by_wkb
        globals()['tap_contamination'] = tap_contamination
        globals()['tap_contamination_by_wkb'] = tap_contamination_by_wkb
        globals()['water_contamination'] = water_contamination
        globals()['water_contamination_by_wkb'] = water_contamination_by_wkb
    finally:
        if _state is not None:
            random.setstate(_state)


def _apply_intervention_D_runtime_globals(facility_payload, inner_payload=None):
    """Rebind the D runtime globals according to the selected D variant.

    The routine can rebind taps and toilets when
    the selected D mode includes WASH facilities. That is necessary for delta_fac
    to affect disease transmission through drinking and defecation pathways.
    """
    if facility_payload is None:
        return
    global dwater_gdf, schools_gdf, markets_gdf, toilets_gdf, drinking_points, school_points, market_points
    global tap_points, tap_tree, toilet_gdf, toilet_geoms, toilet_tree, toilet_usage, POI_POOLS
    global depr_points, depr_weights, inner_polys, inner_tree, MOBILITY_POINTS, MOBILITY_WEIGHTS

    moved_kinds = facility_payload.get('moved_kinds', {}) or {}

    if facility_payload.get('schools_gdf') is not None:
        schools_gdf = facility_payload['schools_gdf'].copy()
    if facility_payload.get('markets_gdf') is not None:
        markets_gdf = facility_payload['markets_gdf'].copy()
    if facility_payload.get('dwater_gdf') is not None:
        dwater_gdf = facility_payload['dwater_gdf'].copy()
    if facility_payload.get('toilets_gdf') is not None:
        toilets_gdf = facility_payload['toilets_gdf'].copy()

    school_points = _clean_geometry_list(facility_payload.get('school_points', []))
    market_points = _clean_geometry_list(facility_payload.get('market_points', []))
    drinking_points = _clean_geometry_list(facility_payload.get('drinking_points', []))
    toilet_geoms = _clean_geometry_list(list(toilets_gdf.geometry)) if toilets_gdf is not None and not getattr(toilets_gdf, 'empty', True) else []
    toilet_gdf = toilets_gdf.copy() if toilets_gdf is not None else _empty_gdf_like(crs=(inner_payload.crs if inner_payload is not None and hasattr(inner_payload, 'crs') else 'EPSG:3857'), extra_cols=['cluster'])
    toilet_tree = STRtree(toilet_geoms) if len(toilet_geoms) else None
    toilet_usage = {getattr(t, 'wkb', None): 0 for t in toilet_geoms}
    tap_points = _clean_geometry_list(list(drinking_points))
    tap_tree = STRtree(tap_points) if len(tap_points) else None

    if bool(moved_kinds.get('water', False)):
        # Facility coordinates remain fixed by the layout seed, while initial
        # contamination is redrawn with the current ABM simulation seed.  This
        # gives baseline/D1 and D2/D3 equivalent stochastic uncertainty without
        # changing the intervention layout or contamination mechanism.
        _refresh_tap_registry_for_current_taps(
            seed=facility_payload.get('simulation_seed', facility_payload.get('seed', None)),
            preserve_existing_flags=False,
        )

    if not isinstance(POI_POOLS, dict):
        POI_POOLS = {}
    payload_pools = facility_payload.get('POI_POOLS', {}) if isinstance(facility_payload.get('POI_POOLS', {}), dict) else {}
    POI_POOLS['generic'] = _clean_geometry_list(list(payload_pools.get('generic', [])))
    POI_POOLS['school'] = _clean_geometry_list(list(payload_pools.get('school', school_points)))
    POI_POOLS['market'] = _clean_geometry_list(list(payload_pools.get('market', market_points)))
    POI_POOLS['taps'] = _clean_geometry_list(list(payload_pools.get('taps', tap_points)))
    POI_POOLS['toilets'] = _clean_geometry_list(list(payload_pools.get('toilets', toilet_geoms)))

    if facility_payload.get('depr_points', None):
        depr_points = _clean_geometry_list(list(facility_payload.get('depr_points', [])))
        depr_weights = None if facility_payload.get('depr_weights', None) is None else np.array(facility_payload.get('depr_weights'), dtype=float)
        MOBILITY_POINTS = list(depr_points)
        MOBILITY_WEIGHTS = None if depr_weights is None else np.array(depr_weights, dtype=float)
        globals()['depr_points'] = list(depr_points)
        globals()['depr_weights'] = None if depr_weights is None else np.array(depr_weights, dtype=float)
        globals()['MOBILITY_POINTS'] = list(MOBILITY_POINTS)
        globals()['MOBILITY_WEIGHTS'] = None if MOBILITY_WEIGHTS is None else np.array(MOBILITY_WEIGHTS, dtype=float)

    if inner_payload is not None and not getattr(inner_payload, 'empty', True):
        inner_polys = _clean_geometry_list(list(inner_payload.geometry))
        inner_tree = STRtree(inner_polys) if len(inner_polys) else None

# ---- diagnostics helper defined before run_scenario to avoid NameError ----
def _build_zone_supply_payload(inner_src, generic_points, generic_weights, school_pts, market_pts, tap_pts, toilet_pts):
    """Diagnostics only: summarise destination supply by zone for gravity-response checks."""
    cols = ["poly_id", "generic_count", "generic_weight", "school_count", "market_count", "tap_count", "toilet_count"]
    if inner_src is None or (hasattr(inner_src, "empty") and inner_src.empty):
        return pd.DataFrame(columns=cols)
    if "poly_id" in inner_src.columns:
        df = inner_src[["poly_id", "geometry"]].copy()
    else:
        df = inner_src.reset_index(drop=True).copy()
        df["poly_id"] = np.arange(len(df))
        df = df[["poly_id", "geometry"]]
    try:
        sidx = df.sindex
    except Exception:
        sidx = None
    poly_geoms = list(df.geometry)
    poly_ids = list(df["poly_id"].astype(int))

    def _pt_to_pid(pt):
        if pt is None:
            return None
        try:
            idxs = list(sidx.intersection(pt.bounds)) if sidx is not None else range(len(poly_geoms))
        except Exception:
            idxs = range(len(poly_geoms))
        for i in idxs:
            try:
                if poly_geoms[int(i)].contains(pt):
                    return poly_ids[int(i)]
            except Exception:
                continue
        return None

    rows = {
        int(pid): {
            "poly_id": int(pid),
            "generic_count": 0,
            "generic_weight": 0.0,
            "school_count": 0,
            "market_count": 0,
            "tap_count": 0,
            "toilet_count": 0,
        }
        for pid in poly_ids
    }

    if generic_points:
        if generic_weights is not None and len(generic_weights) == len(generic_points):
            w = np.array(generic_weights, dtype=float)
        else:
            w = np.ones(len(generic_points), dtype=float)
        for pt, ww in zip(generic_points, w):
            pid = _pt_to_pid(pt)
            if pid is not None:
                rows[int(pid)]["generic_count"] += 1
                rows[int(pid)]["generic_weight"] += float(ww)

    for pts, key in [(school_pts, "school_count"), (market_pts, "market_count"), (tap_pts, "tap_count"), (toilet_pts, "toilet_count")]:
        for pt in (pts or []):
            pid = _pt_to_pid(pt)
            if pid is not None:
                rows[int(pid)][key] += 1

    out = pd.DataFrame(list(rows.values()))
    for c in cols:
        if c not in out.columns:
            out[c] = 0
    return out[cols].sort_values("poly_id").reset_index(drop=True)



def _add_extra_toilets_in_polygon(base_toilets_gdf, target_polygon, extra_fraction=0.25, seed=0):
    """Create extra toilets inside a supplied polygon; returns updated toilets_gdf only."""
    if base_toilets_gdf is None or getattr(base_toilets_gdf, "empty", False) or target_polygon is None:
        return base_toilets_gdf
    try:
        extra_n = int(np.ceil(float(extra_fraction) * float(len(base_toilets_gdf))))
    except Exception:
        extra_n = 0
    if extra_n <= 0:
        return base_toilets_gdf
    try:
        _poly = target_polygon.buffer(0)
    except Exception:
        _poly = target_polygon
    if _poly is None or getattr(_poly, 'is_empty', False):
        return base_toilets_gdf
    _rng_state = random.getstate()
    try:
        random.seed(int(seed) + 404)
        pts = _random_points_in_polygon(_poly, extra_n)
    finally:
        random.setstate(_rng_state)
    if not pts:
        return base_toilets_gdf
    extra_gdf = gpd.GeoDataFrame({
        'feature_type': ['Public Toilet'] * len(pts),
        'cluster': [-1] * len(pts),
    }, geometry=pts, crs=base_toilets_gdf.crs)
    return pd.concat([base_toilets_gdf.copy(), extra_gdf], ignore_index=True)

def _add_extra_taps_in_polygon(base_dwater_gdf, target_polygon, extra_fraction=0.25, seed=0):
    """Create extra drinking-water taps inside a supplied polygon; returns updated dwater_gdf only."""
    if base_dwater_gdf is None or getattr(base_dwater_gdf, "empty", False) or target_polygon is None:
        return base_dwater_gdf
    try:
        extra_n = int(np.ceil(float(extra_fraction) * float(len(base_dwater_gdf))))
    except Exception:
        extra_n = 0
    if extra_n <= 0:
        return base_dwater_gdf
    try:
        _poly = target_polygon.buffer(0)
    except Exception:
        _poly = target_polygon
    if _poly is None or getattr(_poly, 'is_empty', False):
        return base_dwater_gdf
    _rng_state = random.getstate()
    try:
        random.seed(int(seed) + 504)
        pts = _random_points_in_polygon(_poly, extra_n)
    finally:
        random.setstate(_rng_state)
    if not pts:
        return base_dwater_gdf
    extra_gdf = gpd.GeoDataFrame({
        'feature_type': ['Drinking Water'] * len(pts),
        'cluster': [-1] * len(pts),
    }, geometry=pts, crs=base_dwater_gdf.crs)
    return pd.concat([base_dwater_gdf.copy(), extra_gdf], ignore_index=True)

def run_scenario(
    ticks=RUN_TICKS,
    seed=RUN_SEED,
    intervention=None,
    visit_radius_m=1000.0,
    log_nearest_anyway=False,
    save_snapshots=True,
    log_exposures=True,
    track_visits=True,
    show_progress=True,
    # deterministic_tick_rng=True,
):
    """
    ticks: total simulation ticks (hours).
    intervention: None or:
      {'type':'decentralise_centres','extra_hubs': 10, 'seed': 0}
      {'type':'mobility_reduce','rho_mean': 0.1, 'max_radius_m': 3000, 'seed': 0}
    visit_radius_m: distance (m) to count a tap 'visit' (Fig. 3 logging only).
    """

    # Apply the same final-paper epidemiology before every reset.  Scenario
    # differences therefore arise from urban structure/mobility, not hidden
    # changes in contamination fractions or exposure probabilities.
    _epi_overrides = _scenario_epi_overrides(intervention) if '_scenario_epi_overrides' in globals() else {}
    for _k, _v in _epi_overrides.items():
        globals()[_k] = _v
    if _epi_overrides:
        print("[Scenario epidemiology] " + ", ".join(f"{k}={v}" for k, v in _epi_overrides.items()))

    # Reset all disease, resource and mobility state for every scenario.
    # This makes baseline, D1, D2 and D3 independent of execution order.
    reset_agents_and_contamination(seed)

    # Clear cached destination-weight lookup because intervention runs can
    # replace/reorder the POI pool.
    for _cache_attr in ("_w_cache", "_w_index"):
        if hasattr(select_destination, _cache_attr):
            try:
                delattr(select_destination, _cache_attr)
            except Exception:
                pass

    # ─── Apply intervention overrides (A/B) ───────────────────────────────
    global INTERVENTION_HUB_POINTS_GDF, MOVE_MAX_RADIUS_M, INTERVENTION_INNER_GDF, INTERVENTION_OUTER_GDF, depr_points, depr_weights, CURRENT_TICK_EXPOSURE_SOURCE
    global dwater_gdf, schools_gdf, markets_gdf, toilets_gdf, drinking_points, school_points, market_points, POI_POOLS
    global tap_points, tap_tree, toilet_gdf, toilet_geoms, toilet_tree, inner_polys, inner_tree
    global tap_contamination, tap_contamination_by_wkb, water_contamination, water_contamination_by_wkb
    INTERVENTION_HUB_POINTS_GDF = None
    INTERVENTION_INNER_GDF = None
    INTERVENTION_OUTER_GDF = None
    MOVE_MAX_RADIUS_M = None

    typ = None if intervention is None else str(intervention.get("type", ""))
    apply_mobility_params = bool(
        intervention and (typ in ("mobility_reduce", "combined") or intervention.get("apply_mobility_params", False))
    )
    apply_area_redistribute = bool(
        intervention and (typ in ("area_redistribute", "area_redistribute_plus_latrines") or intervention.get("apply_area_redistribute", False))
    )
    globals()["INTERVENTION_D_TARGET_RULE_OPTS"] = {}

    if intervention and intervention.get("seed") is not None:
        random.seed(int(intervention["seed"]))
        np.random.seed(int(intervention["seed"]))

    _depr_points_saved = list(depr_points) if ('depr_points' in globals() and depr_points is not None) else []
    _depr_weights_saved = None if ('depr_weights' not in globals() or depr_weights is None) else np.array(depr_weights, dtype=float).copy()
    _dwater_saved = dwater_gdf.copy() if ('dwater_gdf' in globals() and dwater_gdf is not None) else None
    _schools_saved = schools_gdf.copy() if ('schools_gdf' in globals() and schools_gdf is not None) else None
    _markets_saved = markets_gdf.copy() if ('markets_gdf' in globals() and markets_gdf is not None) else None
    _toilets_saved = toilets_gdf.copy() if ('toilets_gdf' in globals() and toilets_gdf is not None) else None
    _drinking_saved = list(drinking_points) if ('drinking_points' in globals() and drinking_points is not None) else []
    _school_pts_saved = list(school_points) if ('school_points' in globals() and school_points is not None) else []
    _market_pts_saved = list(market_points) if ('market_points' in globals() and market_points is not None) else []
    _poi_pools_saved = dict(POI_POOLS) if ('POI_POOLS' in globals() and isinstance(POI_POOLS, dict)) else None
    _tap_points_saved = list(tap_points) if ('tap_points' in globals() and tap_points is not None) else []
    _toilet_gdf_saved = toilet_gdf.copy() if ('toilet_gdf' in globals() and toilet_gdf is not None) else None
    _toilet_geoms_saved = list(toilet_geoms) if ('toilet_geoms' in globals() and toilet_geoms is not None) else []
    if 'inner_polys' in globals() and inner_polys is not None:
        try:
            if hasattr(inner_polys, 'geometry'):
                _inner_polys_runtime_saved = _clean_geometry_list(list(inner_polys.geometry))
            else:
                _inner_polys_runtime_saved = _clean_geometry_list(list(inner_polys))
        except Exception:
            _inner_polys_runtime_saved = []
    else:
        _inner_polys_runtime_saved = []

    _D_PENDING_PAYLOAD = None
    _D_PENDING_DEPR_POINTS = None
    _D_PENDING_DEPR_WEIGHTS = None
    _D_START_TICK = 0 if apply_area_redistribute else None
    _D_APPLIED = False

    if intervention and intervention.get("type") in ("decentralise_centres", "combined"):
        # add extra hub points using k-means++ style seeding over candidate points
        extra = int(intervention.get("extra_hubs", 0))


        # source candidates: combine POI pools + deprivation points (if available)
        # IMPORTANT: in some city setups POI_POOLS['generic'] can coincide with existing hub centroids.
        # If candidates overlap existing hubs, k-means++ distances become zero and no new hubs can be added.
        cand = []
        if 'POI_POOLS' in globals() and isinstance(POI_POOLS, dict) and 'generic' in POI_POOLS:
            cand.extend(list(POI_POOLS['generic']))
        if 'depr_points' in globals() and depr_points:
            cand.extend(list(depr_points))

        # Ensure candidate points are shapely Points (some pipelines store (x,y) tuples)
        try:
            from shapely.geometry import Point as _Point
            _cand_fixed = []
            for _pt in cand:
                if _pt is None:
                    continue
                if hasattr(_pt, "x") and hasattr(_pt, "y"):
                    _cand_fixed.append(_pt)
                else:
                    try:
                        _cand_fixed.append(_Point(_pt[0], _pt[1]))
                    except Exception:
                        pass
            cand = _cand_fixed
        except Exception:
            pass

        # existing hubs: use centroid_gdf_new if available else activity_feats points
        base_hubs = []
        if 'centroid_gdf_new' in globals() and centroid_gdf_new is not None and (not centroid_gdf_new.empty):
            base_hubs = list(centroid_gdf_new.geometry)
        elif 'activity_feats' in globals() and activity_feats is not None and (not activity_feats.empty):
            base_hubs = list(activity_feats.geometry)

        # De-duplicate candidates and REMOVE candidates that coincide with existing hubs
        try:
            _hub_wkb = set([h.wkb for h in base_hubs if h is not None])
            _seen = set()
            _cand3 = []
            for pt in cand:
                try:
                    w = pt.wkb
                except Exception:
                    continue
                if w in _hub_wkb:
                    continue  # exact same coordinate as an existing hub
                if w in _seen:
                    continue
                _seen.add(w)
                _cand3.append(pt)
            cand = _cand3
        except Exception:
            _hub_wkb = set()

        # If candidate pool is still empty (or tiny), generate extra candidates by sampling within the city boundary.
        # This avoids Intervention A silently becoming a no-op due to candidate/hub overlap.
        try:
            if len(cand) < max(50, 5*int(extra)):
                from shapely.geometry import Point as _Point
                import numpy as _np

                _boundary = None
                if 'outer_gdf' in globals() and outer_gdf is not None and (not outer_gdf.empty):
                    _boundary = outer_gdf.to_crs("EPSG:3857").unary_union
                elif 'gdf_bound_3857' in globals() and gdf_bound_3857 is not None and (not gdf_bound_3857.empty):
                    _boundary = gdf_bound_3857.to_crs("EPSG:3857").unary_union
                elif 'boundary_gdf' in globals() and boundary_gdf is not None and (not boundary_gdf.empty):
                    _boundary = boundary_gdf.to_crs("EPSG:3857").unary_union

                if _boundary is not None and (not _boundary.is_empty):
                    minx, miny, maxx, maxy = _boundary.bounds
                    _rng = _np.random.default_rng(int(intervention.get("seed", 0)) + 991)
                    _need = max(0, max(200, 30*int(extra)) - len(cand))
                    _tries = 0
                    while _need > 0 and _tries < 200000:
                        x = float(_rng.uniform(minx, maxx))
                        y = float(_rng.uniform(miny, maxy))
                        p = _Point(x, y)
                        if p.within(_boundary):
                            try:
                                if p.wkb not in _hub_wkb:
                                    cand.append(p)
                                    _need -= 1
                            except Exception:
                                cand.append(p)
                                _need -= 1
                        _tries += 1
        except Exception:
            pass

        # Debug counts (helps catch "no candidates" early)
        try:
            print(f"[Intervention A DEBUG] candidates={len(cand)} | base_hubs={len(base_hubs)} | requested_extra={extra}")
        except Exception:
            pass

        # guard
        if extra > 0 and cand:
            # dominant centre exclusion (use first hub if available; centroid_gdf_new is expected sorted by weight)
            dominant_hub = base_hubs[0] if len(base_hubs) > 0 else None

            # dominant centre exclusion and minimum separation are derived from the *existing hub spacing*
            # (distribution-based, not an arbitrary km rule).
            def _median_nn_distance(pts):
                if pts is None or len(pts) < 2:
                    return 1500.0
                coords = np.array([(p.x, p.y) for p in pts], dtype=float)
                nn = []
                for i in range(coords.shape[0]):
                    d2 = np.sum((coords - coords[i])**2, axis=1)
                    d2[i] = np.inf
                    nn.append(np.sqrt(np.min(d2)))
                nn = np.asarray(nn, dtype=float)
                nn = nn[np.isfinite(nn) & (nn > 0)]
                return float(np.median(nn)) if nn.size else 1500.0

            d_nn = _median_nn_distance(base_hubs)
            core_excl_m = float(np.clip(1.0 * d_nn, 200.0, 8000.0))   # clamp for safety
            min_sep_m  = float(np.clip(0.75 * d_nn, 150.0, 6000.0))  # clamp for safety

            # --- TUNING: encourage *separated* new hubs ---------------------------------
            # Problem seen in Luanda: new hub areas tend to attach to the dominant inner hub,
            # because most selected points fall inside the dominant outer polygon (same catchment).
            # To keep baseline hubs/boundaries unchanged but still get *distinct* new hub areas,
            # we (a) prefer placing new hubs in *different* baseline outer polygons, and
            # (b) optionally exclude the dominant outer polygon entirely if alternatives exist.
            OUTER_SRC = None
            OUTER_SIDX = None
            try:
                if 'outer_gdf' in globals() and outer_gdf is not None and (not outer_gdf.empty):
                    OUTER_SRC = outer_gdf if str(getattr(outer_gdf, 'crs', None)) == 'EPSG:3857' else outer_gdf.to_crs('EPSG:3857')
                    OUTER_SIDX = OUTER_SRC.sindex
            except Exception:
                OUTER_SRC = None
                OUTER_SIDX = None

            def _outer_idx_for_point(_p):
                """Return index of baseline outer polygon that contains point _p, else None."""
                if OUTER_SRC is None or OUTER_SIDX is None or _p is None:
                    return None
                try:
                    hits = list(OUTER_SIDX.intersection(_p.bounds))
                    for j in hits:
                        if _p.within(OUTER_SRC.geometry.iloc[j]):
                            return int(j)
                except Exception:
                    return None
                return None

            dominant_outer_idx = None
            try:
                if dominant_hub is not None:
                    dominant_outer_idx = _outer_idx_for_point(dominant_hub)
            except Exception:
                dominant_outer_idx = None

            # Apply dominant-centre exclusion if possible
            cand2 = []
            for pt in cand:
                try:
                    if dominant_hub is not None and pt.distance(dominant_hub) <= core_excl_m:
                        continue
                except Exception:
                    pass
                cand2.append(pt)

            # --- NEW: tie candidates to baseline outer polygons and avoid the dominant outer if possible
            cand2_info = [(pt, _outer_idx_for_point(pt)) for pt in cand2]

            # If we can identify the dominant outer polygon, and there are candidates in other outers,
            # drop candidates from the dominant outer to prevent all new hubs clustering around the main hub.
            try:
                if dominant_outer_idx is not None:
                    _has_alt = any((oi is not None and oi != dominant_outer_idx) for _, oi in cand2_info)
                    if _has_alt:
                        cand2_info = [(pt, oi) for (pt, oi) in cand2_info if (oi is None or oi != dominant_outer_idx)]
                        cand2 = [pt for pt, _ in cand2_info]
                        print(f"[Intervention A DEBUG] excluded dominant outer idx={dominant_outer_idx} to encourage separated hubs.")
            except Exception:
                pass

            # If exclusion removed everything, attempt a boundary-based fallback that *still* avoids the dominant outer.
            # Additional hubs do not overlap the dominant outer catchment.
            if not cand2:
                cand2 = []
                # dominant outer polygon (if known)
                try:
                    _dom_poly = OUTER_SRC.geometry.iloc[dominant_outer_idx] if (dominant_outer_idx is not None and OUTER_SRC is not None) else None
                except Exception:
                    _dom_poly = None

                # allowed region = city boundary minus dominant outer (if possible)
                try:
                    _boundary = None
                    if 'outer_gdf' in globals() and outer_gdf is not None and (not outer_gdf.empty):
                        _boundary = outer_gdf.to_crs("EPSG:3857").unary_union
                    elif 'gdf_bound_3857' in globals() and gdf_bound_3857 is not None and (not gdf_bound_3857.empty):
                        _boundary = gdf_bound_3857.to_crs("EPSG:3857").unary_union
                    elif 'boundary_gdf' in globals() and boundary_gdf is not None and (not boundary_gdf.empty):
                        _boundary = boundary_gdf.to_crs("EPSG:3857").unary_union
                except Exception:
                    _boundary = None

                try:
                    if _boundary is not None and (not _boundary.is_empty):
                        _allowed = _boundary if (_dom_poly is None or _dom_poly.is_empty) else _boundary.difference(_dom_poly)
                        if _allowed is not None and (not _allowed.is_empty):
                            minx, miny, maxx, maxy = _allowed.bounds
                            _rng_fb = np.random.default_rng(int(intervention.get("seed", 0)) + 4491)
                            _need = max(200, 30*int(extra))
                            _tries = 0
                            from shapely.geometry import Point as _Point
                            while len(cand2) < _need and _tries < 200000:
                                x = float(_rng_fb.uniform(minx, maxx))
                                y = float(_rng_fb.uniform(miny, maxy))
                                p = _Point(x, y)
                                if p.within(_allowed):
                                    cand2.append(p)
                                _tries += 1
                except Exception:
                    pass

                if not cand2:
                    cand2 = list(cand)
                    try:
                        print("[Intervention A DEBUG] dominant exclusion removed all candidates; falling back to full candidate pool.")
                    except Exception:
                        pass
                else:
                    try:
                        print("[Intervention A DEBUG] dominant exclusion removed all candidates; resampled fallback candidates outside dominant outer.")
                    except Exception:
                        pass

            rng = np.random.default_rng(int(intervention.get("seed", 0)))

            new_hubs = []
            current = list(base_hubs) if base_hubs else []

            # Rebuild candidate->outer mapping after any fallbacks/filters
            cand2_info = [(pt, _outer_idx_for_point(pt)) for pt in cand2]
            used_outer_idxs = set()

            # k-means++ style seeding: p(x) ∝ D(x)^2 where D is distance to nearest existing/added hub
            for _ in range(extra):
                if not cand2:
                    break

                # Prefer placing new hubs in *different* baseline outer polygons where possible.
                eligible = list(range(len(cand2)))
                if OUTER_SRC is not None and OUTER_SIDX is not None and cand2_info:
                    unused = [i for i, (_, oi) in enumerate(cand2_info) if (oi is not None and oi not in used_outer_idxs)]
                    if unused:
                        eligible = unused

                # distances to nearest hub
                dists = np.empty(len(eligible), dtype=float)
                if not current:
                    dists.fill(1.0)
                else:
                    for ii, i in enumerate(eligible):
                        pt = cand2[i]
                        try:
                            dists[ii] = float(min(pt.distance(h) for h in current))
                        except Exception:
                            dists[ii] = 0.0

                # Adaptive relaxation: if min_sep is too strict (all weights zero), progressively relax it.
                _min_sep_try = float(min_sep_m)
                chosen_idx = None
                for _attempt in range(6):
                    d_work = dists.copy()
                    d_work[d_work < _min_sep_try] = 0.0
                    w = d_work**2
                    sw = float(np.sum(w))
                    if np.isfinite(sw) and sw > 0:
                        p = w / sw
                        chosen_local = int(rng.choice(np.arange(len(eligible)), p=p))
                        chosen_idx = int(eligible[chosen_local])
                        if _attempt > 0:
                            try:
                                print(f"[Intervention A DEBUG] relaxed min_sep to {_min_sep_try:.1f}m to enable placement.")
                            except Exception:
                                pass
                        break
                    _min_sep_try *= 0.8  # relax by 20%

                # Final fallback: if still impossible, pick the farthest point (max D) to keep "spread" intent.
                if chosen_idx is None:
                    chosen_idx = int(eligible[int(np.argmax(dists))])
                    try:
                        print("[Intervention A DEBUG] could not satisfy min_sep; selecting farthest candidate by distance instead.")
                    except Exception:
                        pass

                chosen = cand2.pop(chosen_idx)
                # keep cand2_info in sync
                try:
                    chosen_outer = cand2_info.pop(chosen_idx)[1]
                except Exception:
                    chosen_outer = None

                # Avoid exact duplicates (can happen if candidates contained hub points or repeated points)
                try:
                    if current and float(min(chosen.distance(h) for h in current)) < 1e-6:
                        continue
                except Exception:
                    pass

                new_hubs.append(chosen)
                current.append(chosen)
                if chosen_outer is not None:
                    used_outer_idxs.add(chosen_outer)

            # override hub points for metrics + trip attraction (base + new)
            all_hubs = list(base_hubs) + list(new_hubs)
            if all_hubs:
                INTERVENTION_HUB_POINTS_GDF = gpd.GeoDataFrame(
                    geometry=all_hubs,
                    crs=_target_crs if '_target_crs' in globals() else "EPSG:3857"
                )
                # --- NEW (Voronoi hub *sub-areas* for added hubs only; baseline polygons preserved) ---
            # IMPORTANT: We do NOT recompute a citywide Voronoi over (base+new) hubs, because that would
            # change the existing outer/inner boundaries. Instead, we keep baseline inner_gdf/outer_gdf
            # untouched, and create *additional* inner polygons for the new hubs inside the baseline
            # outer polygons (catchments). This makes new hubs visible as extra hub-areas without
            # altering existing hub boundaries.
            try:
                import geopandas as _gpd
                from shapely.geometry import MultiPoint as _MultiPoint
                from shapely.ops import voronoi_diagram as _voronoi_diagram
                from shapely.ops import unary_union as _unary_union
            
                # Baseline polygons (preserve exactly)
                _inner_base = None
                _outer_base = None
                if 'inner_gdf' in globals() and inner_gdf is not None and (not inner_gdf.empty):
                    _inner_base = inner_gdf if str(getattr(inner_gdf, 'crs', None)) == 'EPSG:3857' else inner_gdf.to_crs('EPSG:3857')
                if 'outer_gdf' in globals() and outer_gdf is not None and (not outer_gdf.empty):
                    _outer_base = outer_gdf if str(getattr(outer_gdf, 'crs', None)) == 'EPSG:3857' else outer_gdf.to_crs('EPSG:3857')
            
                if (_inner_base is None) or (_outer_base is None) or (len(new_hubs) == 0):
                    # Nothing to build; keep plotting layers unset (baseline will be used)
                    INTERVENTION_OUTER_GDF = None
                    INTERVENTION_INNER_GDF = None
                else:
                    # Keep baseline outer unchanged
                    INTERVENTION_OUTER_GDF = _outer_base.copy()
            
                    # Spatial index for baseline outers
                    try:
                        _outer_sidx = INTERVENTION_OUTER_GDF.sindex
                    except Exception:
                        _outer_sidx = None
            
                    # Union of baseline inner polygons (used to avoid overlaps)
                    try:
                        _inner_base_union = _inner_base.union_all()
                    except Exception:
                        _inner_base_union = _inner_base.unary_union
            
                    # Assign each new hub to its containing baseline outer polygon (fallback nearest centroid)
                    # Strict rule: do not create new inner polygons inside the dominant outer polygon.
                    _dominant_outer_build = None
                    try:
                        _dom_pt = base_hubs[0] if ('base_hubs' in locals() and base_hubs) else None
                        if _dom_pt is not None and _outer_sidx is not None:
                            _hits = list(_outer_sidx.query(_dom_pt, predicate='contains'))
                            if _hits:
                                _dominant_outer_build = int(_hits[0])
                    except Exception:
                        _dominant_outer_build = None

                    _hub_groups = {}  # outer_idx -> [Point,...]
                    for _pt in list(new_hubs):
                        _oidx = None
                        if _outer_sidx is not None:
                            try:
                                _cands = list(_outer_sidx.query(_pt, predicate='contains'))
                            except Exception:
                                _cands = []
                            if _cands:
                                _oidx = int(_cands[0])
                        if _oidx is None:
                            try:
                                _oidx = int(INTERVENTION_OUTER_GDF.distance(_pt).idxmin())
                            except Exception:
                                _oidx = None
                        if _oidx is None:
                            continue
                        if _dominant_outer_build is not None and int(_oidx) == int(_dominant_outer_build):
                            continue
                        _hub_groups.setdefault(_oidx, []).append(_pt)
            
                    # Parameters for inner area creation (dimensionless ratio; not a km rule)
                    _INNER_MIN_AREA_M2 = 5_000.0
                    # Adaptive sizing: make new hubs larger in larger outer catchments (scale-free).
                    # ratio ranges from MIN..MAX as a function of (outer_area / max_outer_area).
                    _INNER_RATIO_MIN = 0.08
                    _INNER_RATIO_MAX = 0.18
                    try:
                        _MAX_OUTER_AREA = float(INTERVENTION_OUTER_GDF.geometry.area.max())
                    except Exception:
                        _MAX_OUTER_AREA = None

                    # Extra tuning: keep new inner polygons visibly *separate* from baseline inner polygons
                    # by subtracting a small *gap buffer* around baseline inners from the residual space.
                    # The gap is derived from the baseline hub spacing (d_nn), so it is city-scale-relative
                    # and not a fixed distance rule.
                    _INNER_GAP_M = float(np.clip(0.15 * d_nn, 50.0, 1500.0))
            
                    _new_inner_rows = []
                    _used_new_union = None
            
                    for _oidx, _pts in _hub_groups.items():
                        _outer_poly = INTERVENTION_OUTER_GDF.geometry.iloc[_oidx]
                        if _outer_poly is None or _outer_poly.is_empty:
                            continue
            
                        # residual space inside this outer (avoid baseline inner + already placed new inners)
                        _residual = _outer_poly.difference(_inner_base_union.buffer(_INNER_GAP_M))
                        if _used_new_union is not None:
                            _residual = _residual.difference(_used_new_union)
                        if _residual.is_empty:
                            continue
            
                        # Create local Voronoi only among the NEW hubs in this outer polygon
                        if len(_pts) == 1:
                            _cells = [(_pts[0], _residual)]
                        else:
                            _env = _outer_poly.envelope.buffer(1.0)
                            _multi = _MultiPoint(_pts)
                            _vor = _voronoi_diagram(_multi, envelope=_env, edges=False)
                            _polys = [p.buffer(0) for p in getattr(_vor, 'geoms', [])]
                            _cells = []
                            for _poly in _polys:
                                _poly = _poly.intersection(_outer_poly)
                                if _poly.is_empty:
                                    continue
                                # assign to the seed contained by the cell; else nearest centroid
                                _inside = [p for p in _pts if p.within(_poly)]
                                if len(_inside) == 1:
                                    _seed = _inside[0]
                                else:
                                    _c = _poly.centroid
                                    _seed = min(_pts, key=lambda q: q.distance(_c))
                                _cells.append((_seed, _poly.intersection(_residual)))
            
                        # Now carve an 'inner' hub area within each local cell
                        for _seed, _cell in _cells:
                            if _cell is None or _cell.is_empty:
                                continue
                            _A = max(_outer_poly.area, 1.0)
                            if _MAX_OUTER_AREA is not None and _MAX_OUTER_AREA > 0:
                                _s = float(np.clip(_A / _MAX_OUTER_AREA, 0.0, 1.0))
                                _ratio = float(_INNER_RATIO_MIN + (_INNER_RATIO_MAX - _INNER_RATIO_MIN) * (_s ** 0.5))
                            else:
                                _ratio = float(_INNER_RATIO_MIN)
                            _target = max(_INNER_MIN_AREA_M2, _ratio * _A)
                            _r = float(np.sqrt(_target / np.pi))
                            _inner = _cell.intersection(_seed.buffer(_r)).buffer(0)
                            if _inner.is_empty:
                                continue
                            _new_inner_rows.append({'cluster': int(len(_inner_base) + len(_new_inner_rows)), 'geometry': _inner})
                            # update used-new union so next polygons don't overlap
                            _used_new_union = _inner if _used_new_union is None else _used_new_union.union(_inner)
            
                    if _new_inner_rows:
                        _inner_new = _gpd.GeoDataFrame(_new_inner_rows, crs='EPSG:3857')
                        INTERVENTION_INNER_GDF = _gpd.GeoDataFrame(
                            pd.concat([_inner_base.copy(), _inner_new], ignore_index=True),
                            crs='EPSG:3857'
                        )
                    else:
                        INTERVENTION_INNER_GDF = _inner_base.copy()
            except Exception:
                # If anything fails, do not alter baseline plotting layers
                INTERVENTION_OUTER_GDF = None
                INTERVENTION_INNER_GDF = None

            # --- DEBUG: what did Intervention A create? ---
            try:
                _n_new = len(new_hubs) if 'new_hubs' in locals() else 0
                _n_base = len(base_hubs) if 'base_hubs' in locals() else 0
                print(f"[Intervention A DEBUG] base hubs={_n_base} | requested extra={extra} | added new hubs={_n_new}")
                if INTERVENTION_OUTER_GDF is None:
                    print("[Intervention A DEBUG] INTERVENTION_OUTER_GDF=None (will plot baseline outer).")
                else:
                    print(f"[Intervention A DEBUG] INTERVENTION_OUTER_GDF rows={len(INTERVENTION_OUTER_GDF)} "
                          f"area_sum_km2={INTERVENTION_OUTER_GDF.geometry.area.sum()/1e6:.3f}")
                if INTERVENTION_INNER_GDF is None:
                    print("[Intervention A DEBUG] INTERVENTION_INNER_GDF=None (will plot baseline inner).")
                else:
                    _n_inner = len(INTERVENTION_INNER_GDF)
                    _n_base_inner = len(_inner_base) if '_inner_base' in locals() and _inner_base is not None else None
                    _extra_inner = (_n_inner - _n_base_inner) if (_n_base_inner is not None) else None
                    print(f"[Intervention A DEBUG] INTERVENTION_INNER_GDF rows={_n_inner} "
                          + (f"(extra inner={_extra_inner})" if _extra_inner is not None else ""))
                    _areas = np.sort(INTERVENTION_INNER_GDF.geometry.area.to_numpy())
                    if _areas.size:
                        mid = int(0.5*(_areas.size-1))
                        print(f"[Intervention A DEBUG] inner area m^2: min={_areas[0]:.1f} p50={_areas[mid]:.1f} max={_areas[-1]:.1f}")
            except Exception as _e_dbg:
                print("[Intervention A DEBUG] failed:", _e_dbg)

# ------------------------------------------------------------------------
            # broaden exploration pool so dominance/Zipf actually changes via destination choice
            if 'POI_POOLS' in globals() and isinstance(POI_POOLS, dict) and 'generic' in POI_POOLS and new_hubs:
                POI_POOLS['generic'] = list(dict.fromkeys(list(POI_POOLS['generic']) + list(new_hubs)))

    if intervention and apply_area_redistribute:
        _D_MODE = _intervention_d_mode_name(intervention) if '_intervention_d_mode_name' in globals() else str(intervention.get("decentralisation_mode", "all"))
        globals()["INTERVENTION_D_TARGET_RULE_OPTS"] = {
            "use_secondary_score_ranking": bool(intervention.get("use_secondary_score_ranking", False)),
            "secondary_priority_strength": intervention.get("secondary_priority_strength", None),
            "n_secondary_targets": intervention.get("n_secondary_targets", None),
            "n_secondary_targets_frac": intervention.get("n_secondary_targets_frac", None),
            # Additional fields support the optional diagnostic summaries.
            # Main final-paper D uses equalised all-hub redistribution, not secondary-score targeting.
            "inbound_trip_share_by_cluster": intervention.get("inbound_trip_share_by_cluster", globals().get("BASELINE_INBOUND_TRIP_SHARE_BY_CLUSTER", {})),
        }
        try:
            INTERVENTION_OUTER_GDF = outer_gdf.copy() if ('outer_gdf' in globals() and outer_gdf is not None and (not outer_gdf.empty)) else None
            INTERVENTION_INNER_GDF = _build_area_redistributed_inner_gdf(
                inner_gdf.copy() if ('inner_gdf' in globals() and inner_gdf is not None and (not inner_gdf.empty)) else None,
                outer_gdf.copy() if ('outer_gdf' in globals() and outer_gdf is not None and (not outer_gdf.empty)) else None,
                seed=int(intervention.get("seed", seed)),
                k_geom=float(intervention.get("k_geom", 1.0)),
            )
            if INTERVENTION_INNER_GDF is not None and not INTERVENTION_INNER_GDF.empty:
                try:
                    INTERVENTION_HUB_POINTS_GDF = gpd.GeoDataFrame(
                        {"geometry": INTERVENTION_INNER_GDF.geometry.representative_point()},
                        geometry="geometry",
                        crs=INTERVENTION_INNER_GDF.crs,
                    )
                except Exception:
                    INTERVENTION_HUB_POINTS_GDF = None

                _facility_payload = _build_area_redistributed_facilities(
                    INTERVENTION_INNER_GDF,
                    seed=int(intervention.get("seed", seed)),
                    eta=float(intervention.get("eta", 1.90)),
                    beta=float(intervention.get("beta", 0.25)),
                    lambda_blend=float(intervention.get("lambda", intervention.get("lambda_blend", 0.55))),
                    alpha_fac=float(intervention.get("alpha_fac", 0.80)),
                    beta_fac=float(intervention.get("beta_fac", 0.30)),
                    delta_fac=float(intervention.get("delta_fac", 0.0)),
                    delta_dest=float(intervention.get("delta_dest", 0.0)),
                    decentralisation_mode=_D_MODE,
                )
                if _facility_payload is not None:
                    _facility_payload['seed'] = int(intervention.get("seed", seed))
                    _facility_payload['layout_seed'] = int(intervention.get("layout_seed", intervention.get("seed", seed)))
                    _facility_payload['simulation_seed'] = int(seed)
                    _facility_payload['decentralisation_mode'] = _D_MODE
                _D_PENDING_PAYLOAD = _facility_payload
                if _facility_payload is not None and _facility_payload.get('depr_points', None):
                    _D_PENDING_DEPR_POINTS = list(_facility_payload['depr_points'])
                    _D_PENDING_DEPR_WEIGHTS = None if _facility_payload.get('depr_weights', None) is None else np.array(_facility_payload['depr_weights'], dtype=float)
                else:
                    _d_pts, _d_wts = _build_depr_pool_for_area_intervention(
                        INTERVENTION_INNER_GDF,
                        attractiveness_mode=str(intervention.get("attractiveness_mode", "centrality")),
                        eta=float(intervention.get("eta", 2.8)),
                        beta=float(intervention.get("beta", 0.10)),
                        lambda_blend=float(intervention.get("lambda", intervention.get("lambda_blend", 0.55))),
                        delta_dest=float(intervention.get("delta_dest", 0.0)),
                        decentralisation_mode=_D_MODE,
                    )
                    _D_PENDING_DEPR_POINTS = list(_d_pts) if _d_pts else []
                    _D_PENDING_DEPR_WEIGHTS = np.array(_d_wts, dtype=float) if _d_wts is not None else None

                try:
                    _targets_dbg = _select_area_redistribution_targets(INTERVENTION_INNER_GDF)
                except Exception:
                    _targets_dbg = None
                if _targets_dbg is not None:
                    _alloc_dbg = _facility_payload if '_facility_payload' in locals() else None
                    print(
                        "[Intervention D] prepared | active_from_tick=0 | "
                        f"mode={_D_MODE} | "
                        f"area_equalisation_strength={float(_targets_dbg.get('area_equalisation_strength', _targets_dbg.get('shrink_frac_base', 0.0))):.2f} | "
                        f"all secondary hubs targeted={int(_targets_dbg.get('n_possible_secondary_targets', 0))} | "
                        f"dominant area reduction km2={float(_targets_dbg.get('delta_A', 0.0))/1e6:.3f}"
                    )
                    if _alloc_dbg is not None:
                        try:
                            print(
                                "[Intervention D] prepared redistributed totals | "
                                f"taps={int(np.sum(_alloc_dbg.get('alloc_water', [])))} | "
                                f"schools={int(np.sum(_alloc_dbg.get('alloc_school', [])))} | "
                                f"markets={int(np.sum(_alloc_dbg.get('alloc_market', [])))} | "
                                f"toilets={int(np.sum(_alloc_dbg.get('alloc_toilet', [])))}"
                            )
                        except Exception:
                            pass
                # Apply D immediately after reset using baseline destination points with D-only reweighted attraction.
                _apply_intervention_D_runtime_globals(_facility_payload, INTERVENTION_INNER_GDF)
                if typ == "area_redistribute_plus_latrines":
                    try:
                        _baseline_inner = inner_gdf.copy() if ('inner_gdf' in globals() and inner_gdf is not None and (not inner_gdf.empty)) else None
                        _dominant_poly = None
                        if _baseline_inner is not None and not _baseline_inner.empty:
                            _dominant_poly = _baseline_inner.loc[_baseline_inner.geometry.area.idxmax(), "geometry"]
                        if _dominant_poly is None:
                            raise RuntimeError("dominant inner polygon unavailable for Intervention 4 augmentation")
                        if _dominant_poly is not None:
                            toilets_gdf = _add_extra_toilets_in_polygon(toilets_gdf, _dominant_poly, extra_fraction=0.25, seed=int(intervention.get("seed", seed)))
                            dwater_gdf = _add_extra_taps_in_polygon(dwater_gdf, _dominant_poly, extra_fraction=0.25, seed=int(intervention.get("seed", seed)))
                            toilet_gdf = toilets_gdf.copy()
                            toilet_geoms = _clean_geometry_list(list(toilets_gdf.geometry))
                            toilet_tree = STRtree(toilet_geoms) if len(toilet_geoms) else None
                            drinking_points = _clean_geometry_list(list(dwater_gdf.geometry)) if dwater_gdf is not None and not dwater_gdf.empty else []
                            tap_points = _clean_geometry_list(list(drinking_points))
                            tap_tree = STRtree(tap_points) if len(tap_points) else None
                            if not isinstance(POI_POOLS, dict):
                                POI_POOLS = {}
                            POI_POOLS['toilets'] = list(toilet_geoms)
                            POI_POOLS['taps'] = list(tap_points)

                            # Int. 4 hard checks: these additions must be active, not silently D-only.
                            baseline_tap_count = len(_drinking_saved)
                            baseline_toilet_count = len(_toilet_geoms_saved)
                            assert len(drinking_points) > baseline_tap_count, (
                                f"Intervention 4 tap augmentation did not increase taps: "
                                f"{len(drinking_points)} <= {baseline_tap_count}"
                            )
                            assert len(toilet_geoms) > baseline_toilet_count, (
                                f"Intervention 4 toilet augmentation did not increase toilets: "
                                f"{len(toilet_geoms)} <= {baseline_toilet_count}"
                            )
                            print("[INT4 CHECK]", len(drinking_points), len(toilet_geoms))

                            # Register newly added Int. 4 taps as clean and keep previous baseline flags intact.
                            _tap_registry = globals().get('tap_contamination', {})
                            _baseline_wkbs_now = set(getattr(_pt, 'wkb', None) for _pt in _drinking_saved if _pt is not None)
                            for _pt in tap_points:
                                _k = id(_pt)
                                if _k not in _tap_registry:
                                    _tap_registry[_k] = {
                                        'geometry': _pt,
                                        'feces_load': 0.0,
                                        'area': 1.0,
                                        'threshold_total': threshold_per_m2 * 1.0,
                                        'contaminated': False,
                                    }
                                elif getattr(_pt, 'wkb', None) not in _baseline_wkbs_now:
                                    _tap_registry[_k]['feces_load'] = 0.0
                                    _tap_registry[_k]['threshold_total'] = threshold_per_m2 * 1.0
                                    _tap_registry[_k]['contaminated'] = False
                            tap_contamination = _tap_registry
                            tap_contamination_by_wkb = {rec['geometry'].wkb: rec for rec in tap_contamination.values()}
                            water_contamination = tap_contamination
                            water_contamination_by_wkb = tap_contamination_by_wkb
                            globals()['tap_contamination'] = tap_contamination
                            globals()['tap_contamination_by_wkb'] = tap_contamination_by_wkb
                            globals()['water_contamination'] = water_contamination
                            globals()['water_contamination_by_wkb'] = water_contamination_by_wkb
                    except Exception as _e_lat:
                        # Int. 4 is D + extra taps/toilets. For final runs this must fail loudly;
                        # silently continuing would make Int. 4 analytically indistinguishable from D.
                        raise RuntimeError(f"Intervention 4 augmentation failed: {_e_lat}") from _e_lat
                depr_points = list(_D_PENDING_DEPR_POINTS) if _D_PENDING_DEPR_POINTS is not None else list(_depr_points_saved)
                depr_weights = None if _D_PENDING_DEPR_WEIGHTS is None else np.array(_D_PENDING_DEPR_WEIGHTS, dtype=float)
                MOBILITY_POINTS = list(depr_points)
                MOBILITY_WEIGHTS = None if depr_weights is None else np.array(depr_weights, dtype=float)
                globals()['depr_points'] = list(depr_points)
                globals()['depr_weights'] = None if depr_weights is None else np.array(depr_weights, dtype=float)
                globals()['MOBILITY_POINTS'] = list(MOBILITY_POINTS)
                globals()['MOBILITY_WEIGHTS'] = None if MOBILITY_WEIGHTS is None else np.array(MOBILITY_WEIGHTS, dtype=float)
                if not isinstance(POI_POOLS, dict):
                    POI_POOLS = {}
                POI_POOLS['generic'] = list(depr_points)
                _D_APPLIED = True
        except Exception as _e_d:
            if typ == "area_redistribute_plus_latrines":
                raise RuntimeError(f"Intervention 4 setup failed and cannot be used for final analysis: {_e_d}") from _e_d
            print("[Intervention D] failed; reverting to baseline destination pool:", _e_d)
            INTERVENTION_HUB_POINTS_GDF = None
            INTERVENTION_INNER_GDF = None
            INTERVENTION_OUTER_GDF = None
            depr_points = list(_depr_points_saved)
            depr_weights = None if _depr_weights_saved is None else np.array(_depr_weights_saved, dtype=float)
            MOBILITY_POINTS = list(depr_points)
            MOBILITY_WEIGHTS = None if depr_weights is None else np.array(depr_weights, dtype=float)
            dwater_gdf = _dwater_saved.copy() if _dwater_saved is not None else gpd.GeoDataFrame(geometry=[], crs=(inner_gdf.crs if 'inner_gdf' in globals() else 'EPSG:3857'))
            schools_gdf = _schools_saved.copy() if _schools_saved is not None else gpd.GeoDataFrame(geometry=[], crs=(inner_gdf.crs if 'inner_gdf' in globals() else 'EPSG:3857'))
            markets_gdf = _markets_saved.copy() if _markets_saved is not None else gpd.GeoDataFrame(geometry=[], crs=(inner_gdf.crs if 'inner_gdf' in globals() else 'EPSG:3857'))
            toilets_gdf = _toilets_saved.copy() if _toilets_saved is not None else gpd.GeoDataFrame(geometry=[], crs=(inner_gdf.crs if 'inner_gdf' in globals() else 'EPSG:3857'))
            drinking_points = _clean_geometry_list(list(_drinking_saved))
            school_points = _clean_geometry_list(list(_school_pts_saved))
            market_points = _clean_geometry_list(list(_market_pts_saved))
            POI_POOLS = dict(_poi_pools_saved) if isinstance(_poi_pools_saved, dict) else {'generic': list(_depr_points_saved), 'school': list(_school_pts_saved), 'market': list(_market_pts_saved), 'taps': list(_drinking_saved), 'toilets': list(_toilet_geoms_saved)}
            if isinstance(POI_POOLS, dict):
                for _k in list(POI_POOLS.keys()):
                    POI_POOLS[_k] = _clean_geometry_list(list(POI_POOLS.get(_k, [])))
            tap_points = _clean_geometry_list(list(_tap_points_saved))
            tap_tree = STRtree(tap_points) if len(tap_points) else None
            toilet_gdf = _toilet_gdf_saved.copy() if _toilet_gdf_saved is not None else _empty_gdf_like(crs=(inner_gdf.crs if 'inner_gdf' in globals() else 'EPSG:3857'), extra_cols=['cluster'])
            toilet_geoms = _clean_geometry_list(list(_toilet_geoms_saved))
            toilet_tree = STRtree(toilet_geoms) if len(toilet_geoms) else None
            inner_polys = _clean_geometry_list(list(_inner_polys_runtime_saved))
            inner_tree = STRtree(inner_polys) if len(inner_polys) else None
    if intervention and apply_mobility_params:
        # Intervention B (EXTREME TEST): reduce rho strongly, increase gamma, and enforce a radius cap.
        rho_mean = float(intervention.get("rho_mean", 0.02))
        gamma_mean = float(intervention.get("gamma_mean", 0.9))
        MOVE_MAX_RADIUS_M = intervention.get("max_radius_m", None)
        for ag in agent_list:
            try:
                # redraw rho around the new mean, using the same SD as baseline heterogeneity
                ag.exploration_prob = _clip01(np.random.normal(loc=rho_mean, scale=RHO_SD))
                # redraw gamma around the stronger intervention mean
                ag.exploration_decay = max(0.01, float(np.random.normal(loc=gamma_mean, scale=GAMMA_SD)))
            except Exception:
                pass

    total_ticks = int(ticks)
    N = len(agent_list)

    state_hist  = defaultdict(list)
    snapshots   = []
    exposures   = []
    peak_snapshot = None
    peak_tick = 0
    peak_ei_count = -1
    node_vis    = Counter()
    transitions = []   # per-tick counts; incidence uses SE only, EI kept for progression diagnostics

    prev_snap = None

    # --- Tap spatial index (unchanged) --------------------------------------
    taps_gdf = None
    taps_sidx = None
    if 'dwater_gdf' in globals() and dwater_gdf is not None and not dwater_gdf.empty:
        taps_gdf = dwater_gdf if str(dwater_gdf.crs) == "EPSG:3857" else dwater_gdf.to_crs("EPSG:3857")
        try:
            taps_sidx = taps_gdf.sindex
        except Exception:
            taps_sidx = None

    _scenario_tag_runtime = _scenario_tag_from_intervention(intervention)

    def _mean_home_to_nearest(points_local):
        if not points_local:
            return np.nan
        vals = []
        seen_homes = []
        seen_w = set()
        for ag in agent_list:
            h = getattr(ag, 'home_location', None)
            w = getattr(h, 'wkb', None)
            if h is not None and w not in seen_w:
                seen_w.add(w)
                seen_homes.append(h)
            if len(seen_homes) >= 500:
                break
        for h in seen_homes:
            try:
                vals.append(float(min(h.distance(p) for p in points_local if p is not None)))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else np.nan

    def _print_scenario_setup_check():
        try:
            _tap_wkbs_now = set(getattr(p, 'wkb', None) for p in tap_points if p is not None)
            _baseline_tap_wkbs_now = set(getattr(p, 'wkb', None) for p in _drinking_saved if p is not None)
            _added_tap_wkbs = _tap_wkbs_now - _baseline_tap_wkbs_now
            _contam_wkbs = set()
            for rec in globals().get('tap_contamination', {}).values():
                if rec.get('contaminated', False):
                    _contam_wkbs.add(getattr(rec.get('geometry', None), 'wkb', None))
            _clean_added = len([w for w in _added_tap_wkbs if w not in _contam_wkbs])
            print(
                f"[SCENARIO CHECK] {_scenario_tag_runtime} | "
                f"taps={len(tap_points)} toilets={len(toilet_geoms)} | "
                f"contaminated_taps_t0={len(_contam_wkbs)} | clean_added_taps={_clean_added} | "
                f"mean_home_to_tap_m={_mean_home_to_nearest(tap_points):.1f} | "
                f"mean_home_to_toilet_m={_mean_home_to_nearest(toilet_geoms):.1f}"
            )
        except Exception as _e_setup:
            print(f"[SCENARIO CHECK] {_scenario_tag_runtime} failed: {_e_setup}")

    _print_scenario_setup_check()

    # ------------------------------------------------------------------
    # Capture the ACTIVE scenario geometry before globals are restored.
    # These layers are returned with the result so co-occurrence analysis uses
    # the actual D1/D2/D3 plan rather than baseline globals left after the run.
    # ------------------------------------------------------------------
    def _active_points_gdf(points_local, feature_type, weights_local=None):
        _pts = [p for p in (list(points_local) if points_local is not None else []) if p is not None]
        _rows = []
        _weights = None
        if weights_local is not None:
            try:
                _weights = np.asarray(weights_local, dtype=float)
            except Exception:
                _weights = None
        for _i, _pt in enumerate(_pts):
            _row = {
                "feature_id": f"{feature_type}_{_i:05d}",
                "feature_type": feature_type,
                "feature_wkb": getattr(_pt, "wkb", None),
                "geometry": _pt,
            }
            if _weights is not None and _i < len(_weights):
                _row["weight"] = float(_weights[_i])
            else:
                _row["weight"] = 1.0
            _rows.append(_row)
        return gpd.GeoDataFrame(_rows, geometry="geometry", crs="EPSG:3857") if _rows else gpd.GeoDataFrame(
            columns=["feature_id", "feature_type", "feature_wkb", "weight", "geometry"],
            geometry="geometry", crs="EPSG:3857"
        )

    # `depr_points` is the combined mobility pool and therefore contains the
    # true generic POIs plus schools and markets.  Filter the named facilities
    # out for analysis only; destination choice itself is untouched.
    _named_activity_wkbs = {getattr(p, 'wkb', None) for p in list(school_points) + list(market_points) if p is not None}
    _diag_generic_points = []
    _diag_generic_weights = []
    _dw = None if depr_weights is None else np.asarray(depr_weights, dtype=float)
    _seen_generic_wkbs = set()
    for _gi, _gp in enumerate(list(depr_points) if depr_points is not None else []):
        _gwkb = getattr(_gp, 'wkb', None)
        if _gwkb in _named_activity_wkbs or _gwkb in _seen_generic_wkbs:
            continue
        _seen_generic_wkbs.add(_gwkb)
        _diag_generic_points.append(_gp)
        _diag_generic_weights.append(float(_dw[_gi]) if _dw is not None and _gi < len(_dw) else 1.0)
    _diag_generic_weights = np.asarray(_diag_generic_weights, dtype=float)
    if _diag_generic_weights.size:
        _diag_generic_weights[~np.isfinite(_diag_generic_weights)] = 0.0
        _gws = float(_diag_generic_weights.sum())
        _diag_generic_weights = (_diag_generic_weights / _gws) if _gws > 0 else np.full(len(_diag_generic_points), 1.0 / max(1, len(_diag_generic_points)))

    _active_generic_gdf = _active_points_gdf(_diag_generic_points, "generic_activity", _diag_generic_weights)
    _active_schools_gdf = _active_points_gdf(school_points, "school")
    _active_markets_gdf = _active_points_gdf(market_points, "market")
    _active_taps_gdf = _active_points_gdf(tap_points, "tap")
    _active_toilets_gdf = _active_points_gdf(toilet_geoms, "toilet")
    _active_feature_catalog = pd.concat(
        [_active_generic_gdf, _active_schools_gdf, _active_markets_gdf,
         _active_taps_gdf, _active_toilets_gdf],
        ignore_index=True,
    )
    _active_feature_catalog = gpd.GeoDataFrame(
        _active_feature_catalog, geometry="geometry", crs="EPSG:3857"
    ) if not _active_feature_catalog.empty else gpd.GeoDataFrame(
        columns=["feature_id", "feature_type", "feature_wkb", "weight", "geometry"],
        geometry="geometry", crs="EPSG:3857"
    )

    _active_inner_gdf = (
        INTERVENTION_INNER_GDF.copy()
        if INTERVENTION_INNER_GDF is not None and hasattr(INTERVENTION_INNER_GDF, "empty") and not INTERVENTION_INNER_GDF.empty
        else inner_gdf.copy()
    )
    _active_outer_gdf = (
        INTERVENTION_OUTER_GDF.copy()
        if INTERVENTION_OUTER_GDF is not None and hasattr(INTERVENTION_OUTER_GDF, "empty") and not INTERVENTION_OUTER_GDF.empty
        else outer_gdf.copy()
    )

    # Fixed baseline analysis zones are used for cross-scenario co-occurrence.
    _fixed_analysis_zones = vor_gdf.copy() if "vor_gdf" in globals() and vor_gdf is not None and not vor_gdf.empty else outer_gdf.copy()
    if "zone_id" not in _fixed_analysis_zones.columns:
        _fixed_analysis_zones = _fixed_analysis_zones.reset_index(drop=True)
        _fixed_analysis_zones["zone_id"] = np.arange(len(_fixed_analysis_zones), dtype=int)

    # Record the initial cohort so risk analyses can exclude agents seeded E/I.
    _initial_agents_rows = []
    for _ag in agent_list:
        _h = getattr(_ag, "home_location", None)
        _initial_agents_rows.append({
            "agent_id": int(getattr(_ag, "unique_id", -1)),
            "role": str(getattr(_ag, "role", "general")),
            "household_id": int(getattr(_ag, "household_id", -1)),
            "initial_state": str(getattr(_ag, "state", "S")),
            "home_x": float(_h.x) if hasattr(_h, "x") else np.nan,
            "home_y": float(_h.y) if hasattr(_h, "y") else np.nan,
            "rho": float(getattr(_ag, "exploration_prob", np.nan)),
            "gamma": float(getattr(_ag, "exploration_decay", np.nan)),
        })
    _initial_agents_df = pd.DataFrame(_initial_agents_rows)

    # Intervention geometry may use a fixed layout seed. Restore the stochastic
    # simulation RNG to the requested run seed before tick 0 so the 50 ensemble
    # repetitions genuinely differ while using the same urban plan.
    set_seed(int(seed))

    # >>> TRIP FLOWS: build inner polygon lookup (poly_id) and a fast locator
    # Use intervention inner polygons for OD zoning if provided; otherwise baseline inner_gdf.
    _inner_src = INTERVENTION_INNER_GDF if (INTERVENTION_INNER_GDF is not None and hasattr(INTERVENTION_INNER_GDF, 'empty') and not INTERVENTION_INNER_GDF.empty) else (inner_gdf if ('inner_gdf' in globals()) else None)
    inner_ok = (_inner_src is not None and hasattr(_inner_src, 'empty') and (not _inner_src.empty))
    inner_df = None
    inner_sidx = None

    if inner_ok:
        inner_df = _inner_src.copy()
        # ensure a stable poly_id
        if 'poly_id' not in inner_df.columns:
            inner_df = inner_df.reset_index(drop=True).assign(poly_id=lambda d: np.arange(len(d)))

        try:
            inner_sidx = inner_df.sindex
        except Exception:
            inner_sidx = None

        def loc_to_polyid(pt):
            """Return poly_id of inner polygon containing pt, or -1 if none.

            Uses a bbox query plus explicit polygon.contains(point). This avoids
            the fragile point-vs-polygon `predicate="contains"` direction, which
            can return no hits under some GeoPandas/Shapely versions and make
            diagnostic OD zones appear as -1.
            """
            if pt is None:
                return -1
            try:
                if inner_sidx is not None:
                    idxs = list(inner_sidx.intersection(pt.bounds))
                else:
                    idxs = range(len(inner_df))
                for _i in idxs:
                    try:
                        if inner_df.geometry.iloc[int(_i)].contains(pt):
                            return int(inner_df.iloc[int(_i)]['poly_id'])
                    except Exception:
                        continue
            except Exception:
                pass
            return -1
    else:
        def loc_to_polyid(pt):
            return -1

    # agent → starting poly_id (at t=0 right after reset)
    agent_poly = {ag.unique_id: loc_to_polyid(ag.current_location) for ag in agent_list}

    # per-(day, origin, dest) trip counts
    trip_flows = Counter()
    actual_trip_rows = []
    facility_visit_rows = []
    # Diagnostics-only journey state.  A journey starts whenever the selected
    # target changes and accumulates realised movement until that target is
    # reached.  The mobility policy itself is not changed.
    _journey_state = {}

    # ---- Mobility helper (supports start_tick trigger) ----------------------
    def _in_restrict_ring(prev_pt, intr, t):
        if not intr or intr.get('type') != 'mobility_restrict' or center_pt is None:
            return False
        start_tick = int(intr.get('start_tick', 0))
        if t < start_tick:
            return False
        center = intr.get('center', center_pt)
        rad    = float(intr.get('radius_m', 3000.0))
        return prev_pt.distance(center) < rad

    # ---- Interventions executor (each with its own start_tick gate) --------
    def maybe_apply_interventions(t, snap):
        nonlocal _D_APPLIED
        # Intervention D is now applied immediately after reset to avoid a hard Day-2 structural switch.
        return

    # ---- MAIN TICK LOOP (WITH PROGRESS BAR) ---------------------------------
    _desc = f"{CITY_SLUG} sim"
    if intervention and isinstance(intervention, dict) and intervention.get("type"):
        _desc = f"{CITY_SLUG} sim | {intervention.get('type')}"
    _scenario_tag_runtime = _scenario_tag_from_intervention(intervention)

    for t in tqdm(range(total_ticks), desc=_desc, total=total_ticks, leave=True, disable=(not show_progress)):
        CURRENT_TICK_EXPOSURE_SOURCE = {}
        for ag in agent_list:
            # Trip-flow origin polygon at the beginning of the step
            orig_pid = agent_poly.get(ag.unique_id, -1)
            prev_pt_for_diag = ag.current_location if isinstance(ag.current_location, Point) else getattr(ag.current_location, 'centroid', ag.current_location)
            home_pid_for_diag = loc_to_polyid(getattr(ag, 'home_location', None)) if inner_ok else -1

            # Movement-only update.
            # Disease/resource progression and all S→E exposure events are handled once below
            # by run_vectorized_step(snap). Calling ag.step() here would run the older
            # agent-level SEIRS logic as well, double-counting disease and household exposure.
            if _in_restrict_ring(ag.current_location, intervention, t):
                pass  # freeze movement inside restriction ring
            else:
                if 'mobility_logic' in globals():
                    mobility_logic(ag, t % TICKS_PER_DAY)
                else:
                    # Fallback for partial/notebook execution only. In the full script
                    # mobility_logic is defined before run_scenario.
                    ag.step()

            dest_pt_for_diag = ag.current_location if isinstance(ag.current_location, Point) else getattr(ag.current_location, 'centroid', ag.current_location)
            dest_pid_for_diag = loc_to_polyid(dest_pt_for_diag) if inner_ok else -1
            try:
                actual_dist = float(prev_pt_for_diag.distance(dest_pt_for_diag))
            except Exception:
                actual_dist = 0.0
            dest_type = _classify_destination_type(dest_pt_for_diag)

            # Selected-target and completed-journey diagnostics.  The model may
            # reselect a destination on a later tick; in that case a new
            # diagnostics journey begins, exactly reflecting the existing policy.
            _uid_diag = int(getattr(ag, 'unique_id', -1))
            _target_pt = getattr(ag, 'destination', None)
            _target_wkb = getattr(_target_pt, 'wkb', None)
            _target_type = _classify_destination_type(_target_pt) if _target_pt is not None else 'unknown'
            try:
                _selected_distance = float(prev_pt_for_diag.distance(_target_pt)) if _target_pt is not None else np.nan
            except Exception:
                _selected_distance = np.nan
            _js = _journey_state.get(_uid_diag)
            if _target_wkb is None:
                _journey_state.pop(_uid_diag, None)
                _js = None
            if _target_wkb is not None and (_js is None or _js.get('target_wkb') != _target_wkb):
                _js = {
                    'target_wkb': _target_wkb, 'departure_tick': int(t),
                    'departure_x': float(prev_pt_for_diag.x) if hasattr(prev_pt_for_diag, 'x') else np.nan,
                    'departure_y': float(prev_pt_for_diag.y) if hasattr(prev_pt_for_diag, 'y') else np.nan,
                    'distance_m': 0.0,
                }
            if _js is not None:
                _js['distance_m'] = float(_js.get('distance_m', 0.0)) + float(actual_dist)
            _arrived_target = bool(_target_wkb is not None and getattr(dest_pt_for_diag, 'wkb', None) == _target_wkb)
            _journey_distance = float(_js.get('distance_m', actual_dist)) if (_arrived_target and _js is not None) else np.nan
            _travel_time_ticks = int(t - int(_js.get('departure_tick', t)) + 1) if (_arrived_target and _js is not None) else np.nan
            _departure_tick = int(_js.get('departure_tick', t)) if _js is not None else np.nan
            _departure_x = float(_js.get('departure_x', np.nan)) if _js is not None else np.nan
            _departure_y = float(_js.get('departure_y', np.nan)) if _js is not None else np.nan
            if _arrived_target:
                _journey_state.pop(_uid_diag, None)
            elif _js is not None:
                _journey_state[_uid_diag] = _js

            actual_trip_rows.append({
                'scenario': _scenario_tag_runtime,
                'tick': int(t),
                'day': int(t // TICKS_PER_DAY) + 1,
                'agent': int(getattr(ag, 'unique_id', -1)),
                'origin_zone': int(orig_pid),
                'destination_zone': int(dest_pid_for_diag),
                'home_zone': int(home_pid_for_diag),
                'destination_type': dest_type,
                'destination_wkb': getattr(dest_pt_for_diag, 'wkb', None),
                'origin_x': float(prev_pt_for_diag.x) if hasattr(prev_pt_for_diag, 'x') else np.nan,
                'origin_y': float(prev_pt_for_diag.y) if hasattr(prev_pt_for_diag, 'y') else np.nan,
                'destination_x': float(dest_pt_for_diag.x) if hasattr(dest_pt_for_diag, 'x') else np.nan,
                'destination_y': float(dest_pt_for_diag.y) if hasattr(dest_pt_for_diag, 'y') else np.nan,
                'home_x': float(getattr(ag, 'home_location').x) if hasattr(getattr(ag, 'home_location', None), 'x') else np.nan,
                'home_y': float(getattr(ag, 'home_location').y) if hasattr(getattr(ag, 'home_location', None), 'y') else np.nan,
                'role': str(getattr(ag, 'role', 'general')),
                'state_on_move': str(getattr(ag, 'state', 'S')),
                'actual_trip_distance_m': actual_dist,
                'actual_moved_distance_bin': _distance_bin(actual_dist),
                'selected_destination_distance_m': _selected_distance,
                'target_destination_type': _target_type,
                'target_destination_wkb': _target_wkb,
                'arrived_at_selected_destination': _arrived_target,
                'departure_tick': _departure_tick,
                'departure_x': _departure_x,
                'departure_y': _departure_y,
                'journey_distance_m': _journey_distance,
                'travel_time_ticks': _travel_time_ticks,
            })
            if dest_type in ('tap', 'toilet'):
                facility_visit_rows.append({
                    'scenario': _scenario_tag_runtime,
                    'tick': int(t),
                    'day': int(t // TICKS_PER_DAY) + 1,
                    'agent': int(getattr(ag, 'unique_id', -1)),
                    'facility_type': dest_type,
                    'facility_wkb': getattr(dest_pt_for_diag, 'wkb', None),
                    'facility_x': float(dest_pt_for_diag.x) if hasattr(dest_pt_for_diag, 'x') else np.nan,
                    'facility_y': float(dest_pt_for_diag.y) if hasattr(dest_pt_for_diag, 'y') else np.nan,
                    'origin_x': float(prev_pt_for_diag.x) if hasattr(prev_pt_for_diag, 'x') else np.nan,
                    'origin_y': float(prev_pt_for_diag.y) if hasattr(prev_pt_for_diag, 'y') else np.nan,
                    'origin_zone': int(orig_pid),
                    'destination_zone': int(dest_pid_for_diag),
                    'role': str(getattr(ag, 'role', 'general')),
                    'state_on_arrival': str(getattr(ag, 'state', 'S')),
                    'actual_trip_distance_m': actual_dist,
                    'journey_distance_m': _journey_distance if np.isfinite(_journey_distance) else actual_dist,
                    'travel_time_ticks': _travel_time_ticks if np.isfinite(_travel_time_ticks) else 1,
                    'departure_tick': _departure_tick if np.isfinite(_departure_tick) else int(t),
                    'departure_x': _departure_x if np.isfinite(_departure_x) else (float(prev_pt_for_diag.x) if hasattr(prev_pt_for_diag, 'x') else np.nan),
                    'departure_y': _departure_y if np.isfinite(_departure_y) else (float(prev_pt_for_diag.y) if hasattr(prev_pt_for_diag, 'y') else np.nan),
                })

            # visit logging (tap visits) — unchanged
            if track_visits and taps_gdf is not None and taps_gdf.shape[0] > 0:
                loc = ag.current_location
                idxs = []
                if taps_sidx is not None:
                    try:
                        idxs = list(
                            taps_sidx.query(
                                loc.buffer(visit_radius_m).envelope,
                                predicate="intersects"
                            )
                        )
                    except Exception:
                        idxs = []
                if idxs:
                    cand = taps_gdf.iloc[idxs]
                    dists = cand.geometry.distance(loc)
                    nearest_pos  = int(dists.values.argmin())
                    nearest_idx  = cand.index[nearest_pos]
                    nearest_dist = float(dists.iloc[nearest_pos])
                    if nearest_dist <= visit_radius_m or log_nearest_anyway:
                        node_vis[id(taps_gdf.geometry.loc[nearest_idx])] += 1
                else:
                    dists = taps_gdf.geometry.distance(loc)
                    nearest_idx  = int(dists.idxmin())
                    nearest_dist = float(dists.min())
                    if nearest_dist <= visit_radius_m or log_nearest_anyway:
                        node_vis[id(taps_gdf.geometry.loc[nearest_idx])] += 1

            # >>> TRIP FLOWS: destination polygon AFTER stepping
            if inner_ok:
                dest_pid = loc_to_polyid(ag.current_location)
                agent_poly[ag.unique_id] = dest_pid  # update cache

                # count any change of zone (outer<->inner or inner<->inner)
                if dest_pid != orig_pid:
                    day = int(t // TICKS_PER_DAY) + 1  # Day 1..N
                    trip_flows[(day, orig_pid, dest_pid)] += 1

        # snapshot as GeoDataFrame (agents with states & geometry)
        snap = agents_to_gdf(agent_list)

        # disease + environment step
        snap_pre_vectorized = snap.copy()
        if 'run_vectorized_step' in globals():
            snap = run_vectorized_step(snap)

        # Log facility visits that occur inside the vectorized hydration/defecation step.
        try:
            if "_destination_type" in snap.columns:
                _pre_locs = snap_pre_vectorized.set_index("unique_id")["current_location"]
                for _idx, _row in snap.loc[snap["_destination_type"].isin(["tap", "toilet"])].iterrows():
                    _uid = int(_row["unique_id"])
                    _prev_loc = _pre_locs.get(_uid, _row["current_location"])
                    _cur_loc = _row["current_location"]
                    _o_pid = loc_to_polyid(_prev_loc) if inner_ok else -1
                    _d_pid = loc_to_polyid(_cur_loc) if inner_ok else -1
                    try:
                        _dist_v = float(_prev_loc.distance(_cur_loc))
                    except Exception:
                        _dist_v = 0.0
                    facility_visit_rows.append({
                        'scenario': _scenario_tag_runtime,
                        'tick': int(t),
                        'day': int(t // TICKS_PER_DAY) + 1,
                        'agent': _uid,
                        'facility_type': str(_row["_destination_type"]),
                        'facility_wkb': getattr(_cur_loc, 'wkb', None),
                        'facility_x': float(_cur_loc.x) if hasattr(_cur_loc, 'x') else np.nan,
                        'facility_y': float(_cur_loc.y) if hasattr(_cur_loc, 'y') else np.nan,
                        'origin_x': float(_prev_loc.x) if hasattr(_prev_loc, 'x') else np.nan,
                        'origin_y': float(_prev_loc.y) if hasattr(_prev_loc, 'y') else np.nan,
                        'origin_zone': int(_o_pid),
                        'destination_zone': int(_d_pid),
                        'role': str(_row.get('role', 'general')),
                        'state_on_arrival': str(_row.get('state', 'S')),
                        'actual_trip_distance_m': _dist_v,
                        'journey_distance_m': _dist_v,
                        'travel_time_ticks': 1,
                        'departure_tick': int(t),
                        'departure_x': float(_prev_loc.x) if hasattr(_prev_loc, 'x') else np.nan,
                        'departure_y': float(_prev_loc.y) if hasattr(_prev_loc, 'y') else np.nan,
                    })
                    actual_trip_rows.append({
                        'scenario': _scenario_tag_runtime,
                        'tick': int(t),
                        'day': int(t // TICKS_PER_DAY) + 1,
                        'agent': _uid,
                        'origin_zone': int(_o_pid),
                        'destination_zone': int(_d_pid),
                        'home_zone': int(loc_to_polyid(_row.get('home_location', None)) if inner_ok else -1),
                        'destination_type': str(_row["_destination_type"]),
                        'destination_wkb': getattr(_cur_loc, 'wkb', None),
                        'origin_x': float(_prev_loc.x) if hasattr(_prev_loc, 'x') else np.nan,
                        'origin_y': float(_prev_loc.y) if hasattr(_prev_loc, 'y') else np.nan,
                        'destination_x': float(_cur_loc.x) if hasattr(_cur_loc, 'x') else np.nan,
                        'destination_y': float(_cur_loc.y) if hasattr(_cur_loc, 'y') else np.nan,
                        'home_x': float(_row.get('home_location').x) if hasattr(_row.get('home_location', None), 'x') else np.nan,
                        'home_y': float(_row.get('home_location').y) if hasattr(_row.get('home_location', None), 'y') else np.nan,
                        'role': str(_row.get('role', 'general')),
                        'state_on_move': str(_row.get('state', 'S')),
                        'actual_trip_distance_m': _dist_v,
                        'actual_moved_distance_bin': _distance_bin(_dist_v),
                        'selected_destination_distance_m': _dist_v,
                        'target_destination_type': str(_row["_destination_type"]),
                        'target_destination_wkb': getattr(_cur_loc, 'wkb', None),
                        'arrived_at_selected_destination': True,
                        'departure_tick': int(t),
                        'departure_x': float(_prev_loc.x) if hasattr(_prev_loc, 'x') else np.nan,
                        'departure_y': float(_prev_loc.y) if hasattr(_prev_loc, 'y') else np.nan,
                        'journey_distance_m': _dist_v,
                        'travel_time_ticks': 1,
                    })
        except Exception:
            pass

        # Ensure vectorized step did not drop GeoDataFrame metadata
        try:
            import geopandas as _gpd
            if not isinstance(snap, _gpd.GeoDataFrame):
                if "current_location" in snap.columns:
                    snap = _gpd.GeoDataFrame(snap, geometry="current_location", crs="EPSG:3857")
                elif "geometry" in snap.columns:
                    snap = _gpd.GeoDataFrame(snap, geometry="geometry", crs="EPSG:3857")
                    snap["current_location"] = snap.geometry
            else:
                if snap.crs is None:
                    snap = snap.set_crs("EPSG:3857")
                elif str(snap.crs) != "EPSG:3857":
                    snap = snap.to_crs("EPSG:3857")
                if snap.geometry.name != "current_location" and "current_location" in snap.columns:
                    snap = snap.set_geometry("current_location")
                elif "current_location" not in snap.columns and snap.geometry is not None:
                    snap["current_location"] = snap.geometry
        except Exception:
            pass

        if 'update_water_contamination' in globals():
            update_water_contamination(current_tick=t)

        # Household secondary exposure is intentionally NOT applied here.
        # It is already applied once inside run_vectorized_step(). Keeping a second
        # post-vectorized household pass made S→E incidence and prevalence too high
        # and made scenarios appear artificially similar.

        # Persist vectorized disease/resource/location updates back to agent objects.
        # This prevents vectorized state changes from being diagnostic-only.
        try:
            _snap_by_uid = snap.set_index("unique_id")
            _agent_by_uid = {int(getattr(_ag, "unique_id")): _ag for _ag in agent_list}
            for _uid, _row in _snap_by_uid.iterrows():
                _ag = _agent_by_uid.get(int(_uid))
                if _ag is None:
                    continue
                for _col, _attr in [
                    ("state", "state"),
                    ("days_in_state", "days_in_state"),
                    ("hydration_level", "hydration_level"),
                    ("feces_timer", "feces_timer"),
                    ("current_location", "current_location"),
                    ("destination", "destination"),
                ]:
                    if _col in _snap_by_uid.columns:
                        try:
                            setattr(_ag, _attr, _row[_col])
                        except Exception:
                            pass
        except Exception as _e_writeback:
            print("[vectorized writeback] skipped:", _e_writeback)

        # state history (S,E,I,R counts)
        cnt = snap["state"].value_counts()
        for st in ["S", "E", "I", "R"]:
            state_hist[st].append(int(cnt.get(st, 0)))

        # Retain only the peak E+I snapshot for ensemble spatial averaging.
        # This avoids keeping every hourly snapshot for all 50 seeds.
        _ei_now = int(cnt.get("E", 0) + cnt.get("I", 0))
        if _ei_now > peak_ei_count:
            peak_ei_count = _ei_now
            peak_tick = int(t)
            try:
                peak_snapshot = snap[["unique_id", "state", "current_location", "role", "household_id"]].copy()
                if not isinstance(peak_snapshot, gpd.GeoDataFrame):
                    peak_snapshot = gpd.GeoDataFrame(peak_snapshot, geometry="current_location", crs="EPSG:3857")
            except Exception:
                peak_snapshot = snap.copy()

        # per-tick S→E and E→I counts (incidence uses S→E only; E→I is progression)
        SE_count = 0
        EI_count = 0
        if prev_snap is not None:
            prev_states = prev_snap[["unique_id", "state"]].set_index("unique_id")["state"]
            cur_states  = snap[["unique_id", "state"]].set_index("unique_id")["state"]
            aligned = prev_states.to_frame("prev").join(cur_states.to_frame("cur"), how="inner")
            SE_count = int(((aligned["prev"] == "S") & (aligned["cur"] == "E")).sum())
            EI_count = int(((aligned["prev"] == "E") & (aligned["cur"] == "I")).sum())
        transitions.append({"tick": t, "SE": SE_count, "EI": EI_count, "N": N})

        # optional S→E exposure log
        if log_exposures and prev_snap is not None:
            prev = prev_snap[["unique_id", "state"]].set_index("unique_id")
            cur  = snap[["unique_id", "state", "household_id"]].set_index("unique_id")
            j = prev.join(cur, how="inner", lsuffix="_p", rsuffix="_c")
            se = j[(j["state_p"] == "S") & (j["state_c"] == "E")]
            for uid, row in se.iterrows():
                exposures.append({
                    "tick": int(t),
                    "agent": int(uid),
                    "household_id": int(row["household_id"]),
                    "source": CURRENT_TICK_EXPOSURE_SOURCE.get(int(uid), "unknown")
                })

        # --- SNAPSHOT INTEGRITY (ensures plotting/KDE always has locations) ---
        # Some operations may return a plain DataFrame or detach geometry metadata.
        # We re-wrap as a GeoDataFrame in EPSG:3857 and ensure `current_location` is the active geometry.
        try:
            import geopandas as _gpd
            if not isinstance(snap, _gpd.GeoDataFrame):
                if "current_location" in snap.columns:
                    snap = _gpd.GeoDataFrame(snap, geometry="current_location", crs="EPSG:3857")
                elif "geometry" in snap.columns:
                    snap = _gpd.GeoDataFrame(snap, geometry="geometry", crs="EPSG:3857")
                    snap["current_location"] = snap.geometry
                else:
                    # last resort: keep as-is; plotting will skip if no geometry
                    pass
            else:
                # ensure correct CRS + geometry column name
                if snap.crs is None:
                    snap = snap.set_crs("EPSG:3857")
                elif str(snap.crs) != "EPSG:3857":
                    snap = snap.to_crs("EPSG:3857")
                if snap.geometry.name != "current_location" and "current_location" in snap.columns:
                    snap = snap.set_geometry("current_location")
                elif "current_location" not in snap.columns and snap.geometry is not None:
                    snap["current_location"] = snap.geometry
        except Exception:
            pass


        if save_snapshots:
            snapshots.append(snap)

        maybe_apply_interventions(t, snap)
        prev_snap = snap

    # Final spatial state for 50-seed initial/peak/final comparison maps.
    final_snapshot = prev_snap.copy() if prev_snap is not None else None

    # >>> TRIP FLOWS: finalize DataFrames ------------------------------------
    actual_trips_df = pd.DataFrame(actual_trip_rows) if actual_trip_rows else pd.DataFrame(columns=[
        'scenario', 'tick', 'day', 'agent', 'origin_zone', 'destination_zone', 'home_zone',
        'destination_type', 'destination_wkb', 'origin_x', 'origin_y', 'destination_x',
        'destination_y', 'home_x', 'home_y', 'role', 'state_on_move',
        'actual_trip_distance_m', 'actual_moved_distance_bin',
        'selected_destination_distance_m', 'target_destination_type',
        'target_destination_wkb', 'arrived_at_selected_destination',
        'departure_tick', 'departure_x', 'departure_y',
        'journey_distance_m', 'travel_time_ticks'
    ])
    facility_visits_df = pd.DataFrame(facility_visit_rows) if facility_visit_rows else pd.DataFrame(columns=[
        'scenario', 'tick', 'day', 'agent', 'facility_type', 'facility_wkb', 'facility_x',
        'facility_y', 'origin_x', 'origin_y', 'origin_zone', 'destination_zone', 'role',
        'state_on_arrival', 'actual_trip_distance_m', 'journey_distance_m',
        'travel_time_ticks', 'departure_tick', 'departure_x', 'departure_y'
    ])

    # Map WKB coordinates to stable scenario-specific facility IDs.
    _feature_id_by_wkb = {}
    if _active_feature_catalog is not None and not _active_feature_catalog.empty:
        _feature_id_by_wkb = dict(zip(_active_feature_catalog['feature_wkb'], _active_feature_catalog['feature_id']))
    if not actual_trips_df.empty and 'destination_wkb' in actual_trips_df.columns:
        actual_trips_df['destination_id'] = actual_trips_df['destination_wkb'].map(_feature_id_by_wkb)
    else:
        actual_trips_df['destination_id'] = None
    if not facility_visits_df.empty:
        facility_visits_df['facility_id'] = facility_visits_df['facility_wkb'].map(_feature_id_by_wkb)
    else:
        facility_visits_df['facility_id'] = None

    # Remove duplicate facility records produced when the same tap/toilet event
    # is seen by both movement and resource-use diagnostics.
    if not facility_visits_df.empty:
        facility_visits_df = facility_visits_df.sort_values(['tick', 'agent']).drop_duplicates(
            subset=['scenario', 'tick', 'agent', 'facility_type', 'facility_wkb'], keep='last'
        ).reset_index(drop=True)

    # Build a completed amenity visit diary. School/market/generic arrivals are
    # identified only when the current coordinate exactly matches an active
    # destination; tap/toilet rows come from actual resource-use events.
    _catalog_type_by_wkb = dict(zip(_active_feature_catalog['feature_wkb'], _active_feature_catalog['feature_type'])) if not _active_feature_catalog.empty else {}
    _catalog_id_by_wkb = dict(zip(_active_feature_catalog['feature_wkb'], _active_feature_catalog['feature_id'])) if not _active_feature_catalog.empty else {}
    _visit_parts = []
    if not actual_trips_df.empty and 'destination_wkb' in actual_trips_df.columns:
        _activity_visits = actual_trips_df[actual_trips_df['destination_wkb'].isin(_catalog_type_by_wkb)].copy()
        if not _activity_visits.empty:
            _activity_visits['facility_type'] = _activity_visits['destination_wkb'].map(_catalog_type_by_wkb)
            _activity_visits = _activity_visits[_activity_visits['facility_type'].isin(['school', 'market', 'generic_activity'])]
            _activity_visits['facility_id'] = _activity_visits['destination_wkb'].map(_catalog_id_by_wkb)
            _activity_visits['facility_wkb'] = _activity_visits['destination_wkb']
            _activity_visits['facility_x'] = _activity_visits['destination_x']
            _activity_visits['facility_y'] = _activity_visits['destination_y']
            _activity_visits['visit_reason'] = np.where(
                _activity_visits['facility_type'].eq('school'), 'student_school',
                np.where(_activity_visits['facility_type'].eq('market'), 'trader_market', 'routine_activity')
            )
            _activity_visits['state_on_arrival'] = _activity_visits['state_on_move']
            _activity_visits = _activity_visits.sort_values(['agent', 'tick']).drop_duplicates(
                subset=['scenario', 'tick', 'agent', 'facility_id'], keep='last'
            )
            _visit_parts.append(_activity_visits[[
                'scenario','tick','day','agent','role','facility_type','facility_id','facility_wkb',
                'facility_x','facility_y','origin_x','origin_y','home_x','home_y',
                'actual_trip_distance_m','journey_distance_m','travel_time_ticks',
                'departure_tick','departure_x','departure_y','visit_reason','state_on_arrival'
            ]])
    if not facility_visits_df.empty:
        _resource_visits = facility_visits_df.copy()
        _resource_visits['visit_reason'] = np.where(_resource_visits['facility_type'].eq('tap'), 'water_need', 'toilet_need')
        _init_home_lookup = _initial_agents_df.set_index('agent_id')[['home_x','home_y']]
        _resource_visits = _resource_visits.join(_init_home_lookup, on='agent')
        _visit_parts.append(_resource_visits[[
            'scenario','tick','day','agent','role','facility_type','facility_id','facility_wkb',
            'facility_x','facility_y','origin_x','origin_y','home_x','home_y',
            'actual_trip_distance_m','journey_distance_m','travel_time_ticks',
            'departure_tick','departure_x','departure_y','visit_reason','state_on_arrival'
        ]])
    completed_visits_df = pd.concat(_visit_parts, ignore_index=True) if _visit_parts else pd.DataFrame(columns=[
        'scenario','tick','day','agent','role','facility_type','facility_id','facility_wkb',
        'facility_x','facility_y','origin_x','origin_y','home_x','home_y',
        'actual_trip_distance_m','journey_distance_m','travel_time_ticks',
        'departure_tick','departure_x','departure_y','visit_reason','state_on_arrival'
    ])
    if not completed_visits_df.empty:
        completed_visits_df = completed_visits_df.sort_values(['agent','tick','facility_type']).drop_duplicates(
            subset=['scenario','tick','agent','facility_type','facility_id'], keep='last'
        ).reset_index(drop=True)

    # Enrich S→E events with their most defensible location. Tap-water events
    # use the tap visit at the same tick; home/household events use home.
    exposures_df = pd.DataFrame(exposures) if exposures else pd.DataFrame(columns=['tick','agent','household_id','source'])
    if not exposures_df.empty:
        exposures_df['scenario'] = _scenario_tag_runtime
        _home_lookup = _initial_agents_df.set_index('agent_id')[['home_x','home_y','role','initial_state']]
        exposures_df = exposures_df.join(_home_lookup, on='agent')
        exposures_df['exposure_x'] = np.nan
        exposures_df['exposure_y'] = np.nan
        _home_source_mask = exposures_df['source'].isin(['home_water', 'household_secondary'])
        exposures_df.loc[_home_source_mask, 'exposure_x'] = exposures_df.loc[_home_source_mask, 'home_x']
        exposures_df.loc[_home_source_mask, 'exposure_y'] = exposures_df.loc[_home_source_mask, 'home_y']
        exposures_df['exposure_facility_id'] = None
        exposures_df['exposure_facility_wkb'] = None
        if not facility_visits_df.empty:
            _tap_same = facility_visits_df[facility_visits_df['facility_type'].eq('tap')].sort_values(['agent','tick']).drop_duplicates(['agent','tick'], keep='last')
            _tap_same = _tap_same.set_index(['agent','tick'])
            for _i, _erow in exposures_df.loc[exposures_df['source'].eq('tap_water')].iterrows():
                _key = (int(_erow['agent']), int(_erow['tick']))
                if _key in _tap_same.index:
                    _tr = _tap_same.loc[_key]
                    if isinstance(_tr, pd.DataFrame):
                        _tr = _tr.iloc[-1]
                    exposures_df.at[_i, 'exposure_x'] = float(_tr.get('facility_x', np.nan))
                    exposures_df.at[_i, 'exposure_y'] = float(_tr.get('facility_y', np.nan))
                    exposures_df.at[_i, 'exposure_facility_id'] = _tr.get('facility_id', None)
                    exposures_df.at[_i, 'exposure_facility_wkb'] = _tr.get('facility_wkb', None)
        exposures_df['day'] = (pd.to_numeric(exposures_df['tick'], errors='coerce').fillna(0).astype(int) // TICKS_PER_DAY) + 1

        # Previous completed visit and cumulative movement before exposure.
        if not completed_visits_df.empty:
            _vis_sorted = completed_visits_df.sort_values(['agent','tick'])
            _trip_cum = actual_trips_df.sort_values(['agent','tick']).copy()
            _trip_cum['cum_distance_m'] = _trip_cum.groupby('agent')['actual_trip_distance_m'].cumsum()
            _trip_cum['cum_trip_steps'] = _trip_cum.groupby('agent').cumcount() + 1
            for _i, _erow in exposures_df.iterrows():
                _uid = int(_erow['agent']); _t = int(_erow['tick'])
                _prevv = _vis_sorted[(_vis_sorted['agent'] == _uid) & (_vis_sorted['tick'] <= _t)]
                if not _prevv.empty:
                    _pv = _prevv.iloc[-1]
                    exposures_df.at[_i, 'previous_visit_type'] = _pv.get('facility_type', None)
                    exposures_df.at[_i, 'previous_visit_id'] = _pv.get('facility_id', None)
                    exposures_df.at[_i, 'time_since_previous_visit'] = int(_t - int(_pv.get('tick', _t)))
                _prevt = _trip_cum[(_trip_cum['agent'] == _uid) & (_trip_cum['tick'] <= _t)]
                if not _prevt.empty:
                    _pt = _prevt.iloc[-1]
                    exposures_df.at[_i, 'distance_travelled_before_exposure_m'] = float(_pt['cum_distance_m'])
                    exposures_df.at[_i, 'trip_steps_before_exposure'] = int(_pt['cum_trip_steps'])

    if inner_ok and trip_flows:
        flows_rows = [
            {"day": d, "origin_id": o, "dest_id": r, "trips": c}
            for (d, o, r), c in trip_flows.items()
        ]
        trip_flows_df = pd.DataFrame(flows_rows)
        trip_flows_daily = (
            trip_flows_df
            .groupby("day")["trips"]
            .sum()
            .reset_index(name="total_trips")
        )
    else:
        trip_flows_df = pd.DataFrame(columns=["day", "origin_id", "dest_id", "trips"])
        trip_flows_daily = pd.DataFrame(columns=["day", "total_trips"])

    # inner_polys (with poly_id) for later OD / plotting
    inner_polys_return = None
    if inner_ok and inner_df is not None and not inner_df.empty:
        inner_polys_return = inner_df[["poly_id", "geometry"]].copy()

    # diagnostics-only: capture active destination supply seen by the gravity system
    zone_supply_df = _build_zone_supply_payload(
        inner_polys_return if inner_polys_return is not None else inner_df,
        list(_diag_generic_points) if '_diag_generic_points' in locals() else [],
        np.array(_diag_generic_weights, dtype=float) if '_diag_generic_weights' in locals() else None,
        list(school_points) if school_points is not None else [],
        list(market_points) if market_points is not None else [],
        list(tap_points) if tap_points is not None else [],
        list(toilet_geoms) if toilet_geoms is not None else [],
    )

    # diagnostics-only: retain active facility geometry and final contaminated taps for post-run analysis
    tap_points_return = list(tap_points) if tap_points is not None else []
    toilet_geoms_return = list(toilet_geoms) if toilet_geoms is not None else []
    contaminated_tap_wkbs_return = []
    tap_contamination_records_return = pd.DataFrame()
    try:
        _wc_wkb_local = {rec["geometry"].wkb: rec for rec in water_contamination.values()}
        contaminated_tap_wkbs_return = [pt.wkb for pt in tap_points_return if _wc_wkb_local.get(pt.wkb, {}).get("contaminated", False)]

        # Map tap geometries to the stable scenario-specific feature IDs already
        # used by completed-visit and exposure-event logs. No model decisions use
        # any of the fields created below.
        _tap_feature_by_wkb = {}
        if _active_taps_gdf is not None and not getattr(_active_taps_gdf, "empty", True):
            for _, _tr in _active_taps_gdf.iterrows():
                _geom = _tr.geometry
                if _geom is not None:
                    _tap_feature_by_wkb[getattr(_geom, "wkb", None)] = _tr.get("feature_id", None)

        _contam_rows = []
        for _pt in tap_points_return:
            _w = getattr(_pt, "wkb", None)
            _rec = _wc_wkb_local.get(_w, {})
            _initial = bool(_rec.get("initially_contaminated", False))
            _dynamic = bool(_rec.get("dynamically_contaminated", False)) and not _initial
            _final = bool(_rec.get("contaminated", False))
            if _initial:
                _origin = "initial"
                _tick = 0
            elif _dynamic:
                _origin = "dynamic"
                _tick = _rec.get("contamination_tick", None)
            else:
                _origin = "never"
                _tick = None
            _contam_rows.append({
                "tap_id": _tap_feature_by_wkb.get(_w, None),
                "tap_x": float(_pt.x),
                "tap_y": float(_pt.y),
                "initially_contaminated": _initial,
                "dynamically_contaminated": _dynamic,
                "contamination_origin": _origin,
                "contamination_tick": _tick,
                "final_contaminated": _final,
                "final_feces_load": float(_rec.get("feces_load", np.nan)),
                "threshold_total": float(_rec.get("threshold_total", np.nan)),
            })
        tap_contamination_records_return = pd.DataFrame(_contam_rows)
    except Exception as _contam_exc:
        print(f"[Contamination diagnostics] could not build tap provenance table: {_contam_exc}")
        contaminated_tap_wkbs_return = []
        tap_contamination_records_return = pd.DataFrame()

    # restore baseline runtime globals after scenario-specific overrides
    depr_points = list(_depr_points_saved)
    depr_weights = None if _depr_weights_saved is None else np.array(_depr_weights_saved, dtype=float)
    dwater_gdf = _dwater_saved.copy() if _dwater_saved is not None else gpd.GeoDataFrame(geometry=[], crs=(inner_gdf.crs if 'inner_gdf' in globals() else 'EPSG:3857'))
    schools_gdf = _schools_saved.copy() if _schools_saved is not None else gpd.GeoDataFrame(geometry=[], crs=(inner_gdf.crs if 'inner_gdf' in globals() else 'EPSG:3857'))
    markets_gdf = _markets_saved.copy() if _markets_saved is not None else gpd.GeoDataFrame(geometry=[], crs=(inner_gdf.crs if 'inner_gdf' in globals() else 'EPSG:3857'))
    toilets_gdf = _toilets_saved.copy() if _toilets_saved is not None else gpd.GeoDataFrame(geometry=[], crs=(inner_gdf.crs if 'inner_gdf' in globals() else 'EPSG:3857'))
    drinking_points = list(_drinking_saved)
    school_points = list(_school_pts_saved)
    market_points = list(_market_pts_saved)
    POI_POOLS = dict(_poi_pools_saved) if isinstance(_poi_pools_saved, dict) else {'generic': list(_depr_points_saved), 'school': list(_school_pts_saved), 'market': list(_market_pts_saved), 'taps': list(_drinking_saved), 'toilets': list(_toilet_geoms_saved)}
    tap_points = list(_tap_points_saved)
    tap_tree = STRtree(tap_points) if len(tap_points) else None
    toilet_gdf = _toilet_gdf_saved.copy() if _toilet_gdf_saved is not None else gpd.GeoDataFrame(geometry=[], crs=(inner_gdf.crs if 'inner_gdf' in globals() else 'EPSG:3857'))
    toilet_geoms = list(_toilet_geoms_saved)
    toilet_tree = STRtree(toilet_geoms) if len(toilet_geoms) else None
    inner_polys = list(_inner_polys_runtime_saved)
    inner_tree = STRtree(inner_polys) if len(inner_polys) else None

    # Return all data needed by R4, R4b, R4c, helper notebook, etc.
    return {
        "state_history": pd.DataFrame(state_hist),
        "exposures": (
            exposures_df
            if log_exposures else
            pd.DataFrame(columns=["scenario", "tick", "day", "agent", "household_id", "source", "exposure_x", "exposure_y"])
        ),
        "snapshots": snapshots if save_snapshots else [],
        "node_visits": node_vis if track_visits else Counter(),
        "transitions": pd.DataFrame(transitions),   # used for incidence%
        "foi_tick": build_foi_outputs({"transitions": pd.DataFrame(transitions), "state_history": pd.DataFrame(state_hist), "N": N}, ticks_per_day=TICKS_PER_DAY)[0],
        "foi_daily": build_foi_outputs({"transitions": pd.DataFrame(transitions), "state_history": pd.DataFrame(state_hist), "N": N}, ticks_per_day=TICKS_PER_DAY)[1],
        "N": N,
        # OD / mobility outputs:
        "trip_flows": trip_flows_df,
        "trip_flows_daily": trip_flows_daily,
        "actual_trips": actual_trips_df,
        "facility_visits": facility_visits_df,
        "completed_visits": completed_visits_df,
        "initial_agents": _initial_agents_df,
        "peak_snapshot": peak_snapshot,
        "final_snapshot": final_snapshot,
        "peak_tick": int(peak_tick),
        "peak_ei_count": int(max(peak_ei_count, 0)),
        "inner_polys": inner_polys_return,
        "fixed_analysis_zones": _fixed_analysis_zones,
        # Intervention-specific hub geometry (for plots/metrics; baseline=None)
        "hub_points_gdf": INTERVENTION_HUB_POINTS_GDF,
        "inner_gdf_plot": INTERVENTION_INNER_GDF,
        "outer_gdf_plot": INTERVENTION_OUTER_GDF,
        "zone_supply": zone_supply_df,
        "feature_catalog_active": _active_feature_catalog,
        "generic_active": _active_generic_gdf,
        "schools_active": _active_schools_gdf,
        "markets_active": _active_markets_gdf,
        "taps_active": _active_taps_gdf,
        "toilets_active": _active_toilets_gdf,
        "inner_hubs_active": _active_inner_gdf,
        "outer_cells_active": _active_outer_gdf,
        "tap_points_active": tap_points_return,
        "toilet_geoms_active": toilet_geoms_return,
        "contaminated_tap_wkbs": contaminated_tap_wkbs_return,
        "tap_contamination_records": tap_contamination_records_return,
    }

# ===================== Notebook cell 43 =====================
# R3b — STANDARD METRICS + OUTPUT WRITER (NEW; used by baseline + interventions)
# Minimal, self-contained utilities. No refactor into modules.

import os, json, math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import linregress

ANALYSIS_TABLE_PATH = os.path.join(SHARED_OUT, "analysis_table.csv")

def _gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    x = np.sort(x)
    if np.all(x == 0):
        return 0.0
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)

def _entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return float(-np.sum(p * np.log(p)))

def _fit_early_growth(daily_inc, days=(0,1,2)):
    # daily_inc: pd.Series indexed by day with counts (not %)
    y = daily_inc.reindex(list(days)).fillna(0.0).to_numpy(dtype=float)
    # avoid log(0): add 1
    t = np.arange(len(y), dtype=float)
    ly = np.log(y + 1.0)
    if np.all(~np.isfinite(ly)) or np.nanstd(ly) == 0:
        return np.nan
    slope, intercept, r, p, se = linregress(t, ly)
    return float(slope)

def _ensure_inner_centroids(inner_polys_gdf):
    g = inner_polys_gdf.copy()
    if g.crs is None:
        g = g.set_crs("EPSG:3857")
    g["centroid"] = g.geometry.centroid
    return g.set_index("poly_id")["centroid"].to_dict()

def _dist_m(a, b):
    return float(a.distance(b))

def compute_structure_metrics_from_hub_attraction(trip_flows_df, hub_points_gdf):
    '''
    hub_points_gdf: GeoDataFrame with geometry points (candidate hubs), must be in EPSG:3857.
    We compute hub weights as inbound trip counts to nearest hub (by dest centroid).
    '''
    out = {"n_hubs": np.nan, "hub_dominance": np.nan, "hub_zipf_slope": np.nan}
    if trip_flows_df is None or trip_flows_df.empty or hub_points_gdf is None or hub_points_gdf.empty:
        return out

    hubs = hub_points_gdf.copy()
    if hubs.crs is None:
        hubs = hubs.set_crs("EPSG:3857")
    hubs = hubs.to_crs("EPSG:3857")
    hubs = hubs.reset_index(drop=True)
    out["n_hubs"] = int(len(hubs))

    # need dest centroids (poly ids)
    if "dest_id" not in trip_flows_df.columns or "trips" not in trip_flows_df.columns:
        return out

    # map each dest_id -> point (centroid)
    if "INNER_POLY_CENTROIDS" not in globals():
        return out
    centroids = globals()["INNER_POLY_CENTROIDS"]

    dest_ids = trip_flows_df["dest_id"].to_numpy()
    weights = trip_flows_df["trips"].to_numpy(dtype=float)

    # build arrays
    hub_pts = np.array([(p.x, p.y) for p in hubs.geometry])
    hub_w = np.zeros(len(hubs), dtype=float)

    for d, w in zip(dest_ids, weights):
        pt = centroids.get(int(d), None)
        if pt is None:
            continue
        # nearest hub
        dx = hub_pts[:,0] - pt.x
        dy = hub_pts[:,1] - pt.y
        j = int(np.argmin(dx*dx + dy*dy))
        hub_w[j] += float(w)

    tot = hub_w.sum()
    if tot <= 0:
        return out

    out["hub_dominance"] = float(hub_w.max() / tot)

    # Zipf slope on weights (rank 1..K): log(rank) vs log(weight)
    w_sorted = np.sort(hub_w[hub_w>0])[::-1]
    if w_sorted.size >= 2:
        ranks = np.arange(1, w_sorted.size+1, dtype=float)
        slope, intercept, r, p, se = linregress(np.log(ranks), np.log(w_sorted))
        out["hub_zipf_slope"] = float(slope)
    return out

def compute_mobility_metrics(trip_flows_df, inner_centroids, N_agents, ticks, ticks_per_day=24):
    out = {
        "trip_entropy": np.nan,
        "mean_trip_km": np.nan,
        "p90_trip_km": np.nan,
        "trips_per_agent_per_day": np.nan,
        "flow_concentration_top_dest": np.nan,
    }
    if trip_flows_df is None or trip_flows_df.empty:
        return out

    # destination entropy (dest_id)
    dest = trip_flows_df.groupby("dest_id")["trips"].sum()
    p = (dest / dest.sum()).to_numpy(dtype=float)
    out["trip_entropy"] = _entropy(p)

    # length distribution (approx using centroid-to-centroid distance, weighted by trips)
    dists = []
    wts = []
    for _, r in trip_flows_df.iterrows():
        o = inner_centroids.get(int(r["origin_id"]))
        d = inner_centroids.get(int(r["dest_id"]))
        if o is None or d is None:
            continue
        dists.append(_dist_m(o, d)/1000.0)
        wts.append(float(r["trips"]))
    if dists:
        dists = np.asarray(dists, dtype=float)
        wts = np.asarray(wts, dtype=float)
        out["mean_trip_km"] = float(np.average(dists, weights=wts))
        # weighted p90
        order = np.argsort(dists)
        d_sorted = dists[order]; w_sorted = wts[order]
        cw = np.cumsum(w_sorted) / np.sum(w_sorted)
        out["p90_trip_km"] = float(d_sorted[np.searchsorted(cw, 0.9)])
    # trips per agent per day
    total_trips = float(trip_flows_df["trips"].sum())
    days = max(1.0, float(ticks) / float(ticks_per_day))
    out["trips_per_agent_per_day"] = float(total_trips / max(N_agents,1) / days)

    out["flow_concentration_top_dest"] = float(dest.max() / dest.sum()) if dest.sum()>0 else np.nan
    return out

def compute_density_metrics_from_worldpop(boundary_poly, pop_raster_path):
    '''
    Uses the same POP_RASTER as the main pipeline. Returns mean density, CV, gini, top-decile share.
    '''
    out = {"mean_density": np.nan, "density_cv": np.nan, "density_gini": np.nan, "top_decile_pop_share": np.nan}
    try:
        import rasterio
        from rasterio.mask import mask
    except Exception:
        return out

    if boundary_poly is None or pop_raster_path is None or not os.path.exists(pop_raster_path):
        return out

    try:
        with rasterio.open(pop_raster_path) as src:
            geom = [boundary_poly.__geo_interface__]
            arr, _ = mask(src, geom, crop=True, filled=False)
            x = arr[0].astype(float)
            x = x[np.isfinite(x)]
            x = x[x>0]
            if x.size == 0:
                return out
            out["mean_density"] = float(np.mean(x))
            out["density_cv"] = float(np.std(x)/np.mean(x)) if np.mean(x)>0 else np.nan
            out["density_gini"] = _gini(x)
            q90 = float(np.quantile(x, 0.9))
            out["top_decile_pop_share"] = float(x[x>=q90].sum() / x.sum())
    except Exception as e:
        print("[density metrics] failed:", e)
    return out

def compute_water_metrics(boundary_poly, taps_gdf=None, water_polys_gdf=None):
    out = {"waterpoint_density_km2": np.nan, "waterbody_frac": np.nan}
    if boundary_poly is None:
        return out
    area_km2 = float(boundary_poly.area) / 1e6
    if taps_gdf is not None and not taps_gdf.empty and area_km2>0:
        out["waterpoint_density_km2"] = float(len(taps_gdf) / area_km2)
    if water_polys_gdf is not None and not water_polys_gdf.empty and area_km2>0:
        try:
            w = water_polys_gdf.to_crs("EPSG:3857")
            out["waterbody_frac"] = float(w.geometry.area.sum() / boundary_poly.area)
        except Exception:
            pass
    return out

def build_foi_outputs(result, ticks_per_day=24):
    """
    Build tick-level and daily FOI outputs from the true S→E transitions.
    FOI_t = SE_t / S_{t-1}
    Returns a pair: (foi_tick_df, foi_daily_df)
    """
    trans = result.get("transitions", None)
    state_hist = result.get("state_history", None)
    if trans is None or state_hist is None or len(trans) == 0 or len(state_hist) == 0:
        return pd.DataFrame(columns=["tick", "day", "S_prev", "SE", "EI", "N", "foi", "foi_per_100"]),                pd.DataFrame(columns=["day", "SE", "EI", "S_prev_mean", "foi_mean", "foi_peak", "foi_sum"])

    trans_df = trans.copy().reset_index(drop=True)
    state_df = state_hist.copy().reset_index(drop=True)
    if "tick" not in trans_df.columns:
        trans_df["tick"] = np.arange(len(trans_df), dtype=int)
    if "SE" not in trans_df.columns:
        trans_df["SE"] = 0
    if "EI" not in trans_df.columns:
        trans_df["EI"] = 0
    if "N" not in trans_df.columns:
        trans_df["N"] = int(result.get("N", max(1, len(agent_list))))

    state_df = state_df.reindex(range(len(trans_df)))
    S_prev = pd.to_numeric(state_df.get("S", pd.Series(dtype=float)), errors="coerce").shift(1)
    se_tick = pd.to_numeric(trans_df["SE"], errors="coerce")
    ei_tick = pd.to_numeric(trans_df["EI"], errors="coerce")
    foi = se_tick / S_prev.replace({0: np.nan})
    foi = foi.replace([np.inf, -np.inf], np.nan)

    foi_tick_df = pd.DataFrame({
        "tick": pd.to_numeric(trans_df["tick"], errors="coerce").fillna(0).astype(int),
        "S_prev": S_prev,
        "SE": se_tick,
        "EI": ei_tick,
        "N": pd.to_numeric(trans_df["N"], errors="coerce"),
        "foi": foi,
    })
    foi_tick_df["day"] = (foi_tick_df["tick"] // ticks_per_day).astype(int)
    foi_tick_df["foi_per_100"] = 100.0 * foi_tick_df["foi"]

    foi_daily_df = (
        foi_tick_df.groupby("day", dropna=False)
        .agg(
            SE=("SE", "sum"),
            EI=("EI", "sum"),
            S_prev_mean=("S_prev", "mean"),
            foi_mean=("foi", "mean"),
            foi_peak=("foi", "max"),
            foi_sum=("foi", "sum"),
        )
        .reset_index()
    )
    return foi_tick_df, foi_daily_df


def compute_epidemic_metrics(result, inner_centroids, hub_points_for_dispersion=None, ticks_per_day=24):
    out = {
        "early_growth": np.nan,
        "peak_incidence_day": np.nan,
        "time_to_peak_day": np.nan,
        "attack_rate_7d": np.nan,
        "total_cases": np.nan,
        "final_attack_rate": np.nan,
        "persistence_index": np.nan,
        "foi_mean": np.nan,
        "foi_peak": np.nan,
        "r90_from_dominant_hub_km": np.nan,
        "infected_area_frac_peak": np.nan,
    }
    trans = result.get("transitions", None)
    if trans is None or len(trans)==0:
        return out
    N = int(result.get("N", max(1, len(agent_list))))
    trans = trans.copy()
    foi_tick_df, foi_daily_df = build_foi_outputs(result, ticks_per_day=ticks_per_day)
    if foi_tick_df is not None and not foi_tick_df.empty:
        foi_vals = pd.to_numeric(foi_tick_df["foi"], errors="coerce").iloc[1:]
        out["foi_mean"] = float(np.nanmean(foi_vals.values)) if len(foi_vals) else np.nan
        out["foi_peak"] = float(np.nanmax(foi_vals.values)) if len(foi_vals) else np.nan
    trans["day"] = (trans["tick"] // ticks_per_day).astype(int)
    daily = trans.groupby("day")["SE"].sum().astype(float)  # new infections only (S→E)
    out["early_growth"] = _fit_early_growth(daily, days=(0,1,2))
    peak = float(daily.max()) if len(daily)>0 else np.nan
    out["peak_incidence_day"] = peak
    if np.isfinite(peak) and peak>0:
        out["time_to_peak_day"] = float(daily.idxmax())
    out["attack_rate_7d"] = float(daily.reindex(range(0,7)).fillna(0).sum() / max(N,1))
    out["total_cases"] = float(daily.sum()) if len(daily)>0 else np.nan
    out["final_attack_rate"] = float(out["total_cases"] / max(N,1)) if np.isfinite(out["total_cases"]) else np.nan
    # persistence: mean incidence in last 2 days / peak
    tail = float(daily.reindex([max(daily.index.max()-1,0), daily.index.max()]).fillna(0).mean()) if len(daily)>0 else np.nan
    out["persistence_index"] = float(tail/peak) if np.isfinite(peak) and peak>0 else np.nan

    # spatial dispersion at peak day from dominant hub: r90 + infected area fraction
    snaps = result.get("snapshots", [])
    if not snaps:
        return out
    peak_day = int(out["time_to_peak_day"]) if np.isfinite(out["time_to_peak_day"]) else None
    if peak_day is None:
        return out
    # pick snapshot closest to end of peak day
    peak_tick = int((peak_day+1)*ticks_per_day - 1)
    snap = min(snaps, key=lambda s: abs(int(s.get("tick",0)) - peak_tick))
    
    snap_obj = snap
    inf_pts = []
    try:
        if isinstance(snap_obj, gpd.GeoDataFrame):
            if "state" in snap_obj.columns:
                gI = snap_obj.loc[snap_obj["state"].isin(["E", "I"])]
            else:
                gI = snap_obj
            inf_pts = list(gI.geometry)
        elif isinstance(snap_obj, dict):
            inf_pts = [Point(xy) for xy in snap_obj.get("infected_xy", [])]
    except Exception:
        inf_pts = []

    if not inf_pts:
        return out

    # infected area fraction: map infected points to poly_id by nearest centroid
    # approximate: nearest centroid id
    cent_items = list(inner_centroids.items())
    ids = np.array([k for k,_ in cent_items], dtype=int)
    pts = np.array([(p.x,p.y) for _,p in cent_items], dtype=float)
    infected_polys=set()
    for p in inf_pts:
        dx=pts[:,0]-p.x; dy=pts[:,1]-p.y
        infected_polys.add(int(ids[int(np.argmin(dx*dx+dy*dy))]))
    out["infected_area_frac_peak"] = float(len(infected_polys)/max(len(inner_centroids),1))

    # dominant hub point
    if hub_points_for_dispersion is not None and not hub_points_for_dispersion.empty:
        hubs=hub_points_for_dispersion.to_crs("EPSG:3857")
        # choose by inbound weights computed earlier if available; else first
        hub_pt = hubs.geometry.iloc[0]
        d = np.array([hub_pt.distance(p) for p in inf_pts], dtype=float)/1000.0
        out["r90_from_dominant_hub_km"] = float(np.quantile(d, 0.9))
    return out

def append_analysis_row(row_dict):
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(ANALYSIS_TABLE_PATH):
        try:
            df = pd.read_csv(ANALYSIS_TABLE_PATH)
            df = pd.concat([df, df_row], ignore_index=True)
        except Exception:
            df = df_row
    else:
        df = df_row
    df.to_csv(OUT_DIR /ANALYSIS_TABLE_PATH, index=False)

# ===================== Notebook cell 44 =====================
# R4 — BASELINE RUN + EXPORTS (+ spread radius at peak-I day)
assert 'CITY_SLUG' in globals() and 'SHARED_OUT' in globals(), "Run R0a/R0b first."

import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm

# --- Helper: rebuild OD flows from snapshots if trip_flows is empty ----------
def _build_od_from_snapshots(result, inner_gdf, ticks_per_day=None):
    """
    Rebuild OD flows from snapshots + inner_gdf.
    Returns:
      trip_flows_df: columns [day, origin_id, dest_id, trips]
      trip_flows_daily: [day, total_trips]
      inner_polys: GeoDataFrame [poly_id, geometry] in EPSG:3857
    """
    if ticks_per_day is None:
        ticks_per_day = globals().get("TICKS_PER_DAY", 24)

    snaps = result.get("snapshots", [])
    if not snaps:
        print("[R4/_build_od] No snapshots in result; cannot rebuild OD flows.")
        empty_tf  = pd.DataFrame(columns=["day","origin_id","dest_id","trips"])
        empty_tfd = pd.DataFrame(columns=["day","total_trips"])
        return empty_tf, empty_tfd, None

    if 'inner_gdf' not in globals() or inner_gdf is None or inner_gdf.empty:
        print("[R4/_build_od] inner_gdf missing/empty; cannot rebuild OD flows.")
        empty_tf  = pd.DataFrame(columns=["day","origin_id","dest_id","trips"])
        empty_tfd = pd.DataFrame(columns=["day","total_trips"])
        return empty_tf, empty_tfd, None

    # Standardise inner polygons to EPSG:3857 + poly_id
    inner = inner_gdf.copy()
    inner = inner.to_crs("EPSG:3857")
    if "poly_id" not in inner.columns:
        inner = inner.reset_index().rename(columns={"index": "poly_id"})
    inner = gpd.GeoDataFrame(inner[["poly_id","geometry"]],
                             geometry="geometry", crs="EPSG:3857")

    rows = []
    TPD = int(ticks_per_day)

    for t in tqdm(range(1, len(snaps)), desc=f"{CITY_SLUG} OD rebuild", total=len(snaps)-1, leave=False):
        day_t = (t // TPD) + 1

        # previous tick
        g_prev = snaps[t - 1][["unique_id", "current_location"]].copy()
        g_prev = gpd.GeoDataFrame(g_prev, geometry="current_location", crs="EPSG:3857")
        g_prev = gpd.sjoin(
            g_prev, inner, predicate="within", how="left"
        )[["unique_id", "poly_id"]].rename(columns={"poly_id": "o"})

        # current tick
        g_cur = snaps[t][["unique_id", "current_location"]].copy()
        g_cur = gpd.GeoDataFrame(g_cur, geometry="current_location", crs="EPSG:3857")
        g_cur = gpd.sjoin(
            g_cur, inner, predicate="within", how="left"
        )[["unique_id", "poly_id"]].rename(columns={"poly_id": "d"})

        j = g_prev.merge(g_cur, on="unique_id", how="inner")
        j = j[(j["o"].notna()) & (j["d"].notna())]
        if j.empty:
            continue

        cnt = (
            j.groupby(["o", "d"])
             .size()
             .reset_index(name="trips")
             .rename(columns={"o":"origin_id","d":"dest_id"})
        )
        cnt["day"] = day_t
        rows.append(cnt)

    if rows:
        trip_flows_df = pd.concat(rows, ignore_index=True)[
            ["day","origin_id","dest_id","trips"]
        ]
    else:
        trip_flows_df = pd.DataFrame(columns=["day","origin_id","dest_id","trips"])

    if not trip_flows_df.empty:
        trip_flows_daily = (
            trip_flows_df.groupby("day")["trips"]
            .sum().reset_index(name="total_trips")
        )
    else:
        trip_flows_daily = pd.DataFrame(columns=["day","total_trips"])

    return trip_flows_df, trip_flows_daily, inner[["poly_id","geometry"]]

# --- Run baseline scenario ----------------------------------------------------
base = run_scenario(ticks=RUN_TICKS, seed=RUN_SEED, intervention=None)
print("keys:", base.keys())
print("transitions type:", type(base.get("transitions", None)))
print("transitions shape:", None if base.get("transitions", None) is None else base["transitions"].shape)


# --- Incidence (%) per tick from transitions (new infections only) -------
trans_df = base.get("transitions", None)
N = base.get("N", len(agent_list))
if trans_df is None or len(trans_df) == 0:
    raise RuntimeError("No transitions recorded. Make sure you re-ran R3 before R4.")

# tick incidence% = 100/N * SE (new S→E infections only)
inc_pct_tick = 100.0 * trans_df["SE"].to_numpy() / max(N, 1)
inc_tick_series = pd.Series(inc_pct_tick, index=trans_df["tick"].to_numpy(), name="inc")

# Export baseline incidence (tick-level) with CITY_SLUG tag
os.makedirs(SHARED_OUT, exist_ok=True)
pd.DataFrame({"tick": inc_tick_series.index, "inc": inc_tick_series.values}).to_csv(
    os.path.join(SHARED_OUT, f"{CITY_SLUG}_incidence.csv"), index=False
)

# Also keep base_inc (for helper notebook that expects *_base_inc.csv)
pd.DataFrame({"tick": inc_tick_series.index, "inc": inc_tick_series.values}).to_csv(
    os.path.join(SHARED_OUT, f"{CITY_SLUG}_base_inc.csv"), index=False
)

# FOI exports (tick-level and daily) from true S→E transitions
foi_tick_base, foi_daily_base = build_foi_outputs(base, ticks_per_day=TICKS_PER_DAY)
foi_tick_base.to_csv(
    os.path.join(SHARED_OUT, f"{CITY_SLUG}_foi_tick.csv"), index=False
)
foi_daily_base.to_csv(
    os.path.join(SHARED_OUT, f"{CITY_SLUG}_foi_daily.csv"), index=False
)

# Keep state_history for later with CITY_SLUG tag
base["state_history"].to_csv(
    os.path.join(SHARED_OUT, f"{CITY_SLUG}_state_history.csv"), index=False
)

# --- TRUE SPREAD RADIUS at peak-I day (80th percentile) — old method ---------
def compute_spread_radius_km_at_peak_I(result, center_point):
    if center_point is None:
        return 0.0
    I = result["state_history"]["I"].to_numpy()
    pk_tick = int(np.argmax(I)) if len(I) else 0
    peak_day = pk_tick // TICKS_PER_DAY
    t0 = peak_day * TICKS_PER_DAY
    t1 = min((peak_day + 1) * TICKS_PER_DAY, len(result["snapshots"]))
    pts = []
    for t in range(t0, t1):
        snap = result["snapshots"][t]
        pts.extend(list(snap[snap["state"].isin(["E", "I"])].geometry))
    if not pts:
        return 0.0
    d = np.array([p.distance(center_point) for p in pts], dtype=float)
    return float(np.quantile(d, 0.8) / 1000.0)

spread_km = compute_spread_radius_km_at_peak_I(base, center_pt)

# Save spread JSON with CITY_SLUG tag
with open(os.path.join(SHARED_OUT, f"{CITY_SLUG}_spread.json"), "w") as f:
    json.dump({"radius_km": spread_km}, f)

print(f"[{CITY_SLUG}] Baseline exported: incidence% per tick → {CITY_SLUG}_incidence.csv")
print(f"[{CITY_SLUG}] spread radius (80% at peak-I day): {spread_km:.2f} km")

# --- TRIP FLOWS: use R3 outputs IF present, else rebuild from snapshots ------
tf  = base.get("trip_flows", pd.DataFrame())
tfd = base.get("trip_flows_daily", pd.DataFrame())
inner_polys = base.get("inner_polys", None)

if tf is None or tf.empty or tfd is None or tfd.empty:
    print(f"[{CITY_SLUG}] trip_flows empty or missing → rebuilding from snapshots.")
    tf, tfd, inner_polys = _build_od_from_snapshots(base, inner_gdf, TICKS_PER_DAY)
else:
    print(f"[{CITY_SLUG}] Using trip_flows from run_scenario (R3).")

# Save OD flows if available
if tf is not None and not tf.empty:
    tf_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_tripflows.csv")
    tf.to_csv(OUT_DIR /tf_path, index=False)
    print(f"[{CITY_SLUG}] Saved trip flows → {tf_path}")
else:
    print(f"[{CITY_SLUG}] WARNING: trip_flows remains empty; no *_tripflows.csv written.")

if tfd is not None and not tfd.empty:
    tfd_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_tripflows_daily.csv")
    tfd.to_csv(OUT_DIR /tfd_path, index=False)
    print(f"[{CITY_SLUG}] Saved daily trip flows → {tfd_path}")
else:
    print(f"[{CITY_SLUG}] WARNING: trip_flows_daily remains empty; no *_tripflows_daily.csv written.")

# optional: attach polygon centroids for convenience
if inner_polys is not None and isinstance(inner_polys, gpd.GeoDataFrame) and not inner_polys.empty:
    inner_poly_out = inner_polys.to_crs("EPSG:3857").copy()
    inner_poly_out = inner_poly_out.assign(
        x=lambda d: d.geometry.centroid.x,
        y=lambda d: d.geometry.centroid.y
    )[["poly_id","x","y"]]
    inner_poly_out.to_csv(
        os.path.join(SHARED_OUT, f"{CITY_SLUG}_innerpoly_centroids.csv"),
        index=False
    )
    print(f"[{CITY_SLUG}] Saved inner polygon centroids → {CITY_SLUG}_innerpoly_centroids.csv")
else:
    print(f"[{CITY_SLUG}] No inner_polys to save centroids for.")


def _build_baseline_inbound_trip_share_by_cluster(trip_flows_df, inner_src=None, baseline_inner=None):
    """Return inbound trip-share map keyed by poly_id, cluster, and row index.

    The new Intervention D secondary-score rule uses this map so secondary hubs
    can be ranked by realised baseline inbound trips as well as hub area.
    The function is deliberately defensive because different parts of the script
    carry either poly_id, cluster, or just row order.
    """
    out = {}
    if trip_flows_df is None or getattr(trip_flows_df, "empty", True):
        return out
    if "dest_id" in trip_flows_df.columns:
        dest_col = "dest_id"
    elif "destination_zone" in trip_flows_df.columns:
        dest_col = "destination_zone"
    else:
        return out
    trips_col = "trips" if "trips" in trip_flows_df.columns else None
    if trips_col is None:
        return out
    df = trip_flows_df[[dest_col, trips_col]].copy()
    df[dest_col] = pd.to_numeric(df[dest_col], errors="coerce")
    df[trips_col] = pd.to_numeric(df[trips_col], errors="coerce").fillna(0.0)
    df = df[np.isfinite(df[dest_col]) & (df[dest_col] >= 0)].copy()
    if df.empty or float(df[trips_col].sum()) <= 0:
        return out
    agg = df.groupby(dest_col, as_index=False)[trips_col].sum()
    total = float(agg[trips_col].sum())

    # Direct poly_id/index keys from the OD table.
    for _, r in agg.iterrows():
        pid = int(r[dest_col])
        share = float(r[trips_col]) / total if total > 0 else 0.0
        for k in (pid, str(pid), f"poly_id:{pid}", f"index:{pid}"):
            out[k] = share

    # If the baseline inner polygons include clusters, also create cluster keys.
    lookup = None
    if isinstance(inner_src, gpd.GeoDataFrame) and not inner_src.empty:
        lookup = inner_src.copy().reset_index(drop=True)
    if (lookup is None or lookup.empty or "cluster" not in lookup.columns) and isinstance(baseline_inner, gpd.GeoDataFrame) and not baseline_inner.empty:
        lookup = baseline_inner.copy().reset_index(drop=True)
    if lookup is not None and not lookup.empty:
        if "poly_id" not in lookup.columns:
            lookup["poly_id"] = np.arange(len(lookup), dtype=int)
        if "cluster" in lookup.columns:
            tmp = agg.rename(columns={dest_col: "poly_id"}).copy()
            tmp["poly_id"] = tmp["poly_id"].astype(int)
            lk = lookup[["poly_id", "cluster"]].copy()
            lk["poly_id"] = pd.to_numeric(lk["poly_id"], errors="coerce").astype("Int64")
            merged = tmp.merge(lk, on="poly_id", how="left")
            if not merged.empty and "cluster" in merged.columns:
                cagg = merged.dropna(subset=["cluster"]).groupby("cluster", as_index=False)[trips_col].sum()
                for _, r in cagg.iterrows():
                    try:
                        cid = int(r["cluster"])
                    except Exception:
                        cid = r["cluster"]
                    share = float(r[trips_col]) / total if total > 0 else 0.0
                    for k in (cid, str(cid), f"cluster:{cid}"):
                        out[k] = share
    return out


BASELINE_INBOUND_TRIP_SHARE_BY_CLUSTER = _build_baseline_inbound_trip_share_by_cluster(
    tf,
    inner_src=inner_polys if isinstance(inner_polys, gpd.GeoDataFrame) else None,
    baseline_inner=inner_gdf if 'inner_gdf' in globals() else None,
)
print(f"[D target rule] baseline inbound-trip share keys={len(BASELINE_INBOUND_TRIP_SHARE_BY_CLUSTER)}")


# Make inner polygon centroids available for downstream metric computations
if inner_polys is not None and not inner_polys.empty:
    INNER_POLY_CENTROIDS = _ensure_inner_centroids(inner_polys)
else:
    INNER_POLY_CENTROIDS = {}


# ─── STANDARD METRICS ROW (baseline) ───────────────────────────────────────
# load trip flows (tf) from memory or from file
trip_flows_df = tf if 'tf' in globals() else None
inner_centroids = INNER_POLY_CENTROIDS

# choose hub points for attraction-based structure: use cluster centroids if available else activity_feats points
hub_pts_for_metrics = None
if 'centroid_gdf_new' in globals() and centroid_gdf_new is not None and not centroid_gdf_new.empty:
    hub_pts_for_metrics = centroid_gdf_new[["geometry"]].copy()
    hub_pts_for_metrics = hub_pts_for_metrics.set_crs(activity_feats.crs if 'activity_feats' in globals() else "EPSG:3857")
elif 'activity_feats' in globals() and activity_feats is not None and not activity_feats.empty:
    hub_pts_for_metrics = activity_feats[["geometry"]].copy()

struct_m = compute_structure_metrics_from_hub_attraction(trip_flows_df, hub_pts_for_metrics)
mob_m = compute_mobility_metrics(trip_flows_df, inner_centroids, N_agents=N, ticks=RUN_TICKS, ticks_per_day=TICKS_PER_DAY)

# boundary for density/water metrics
_boundary_poly = None
if 'boundary_gdf' in globals() and boundary_gdf is not None and not boundary_gdf.empty:
    _boundary_poly = boundary_gdf.to_crs("EPSG:3857").unary_union
elif 'gdf_bound_3857' in globals() and gdf_bound_3857 is not None and not gdf_bound_3857.empty:
    _boundary_poly = gdf_bound_3857.to_crs("EPSG:3857").unary_union

dens_m = compute_density_metrics_from_worldpop(_boundary_poly, POP_RASTER if 'POP_RASTER' in globals() else None)
water_m = compute_water_metrics(_boundary_poly, taps_gdf=taps_gdf if 'taps_gdf' in globals() else None,
                               water_polys_gdf=water_gdf if 'water_gdf' in globals() else None)

epi_m = compute_epidemic_metrics(base, inner_centroids, hub_points_for_dispersion=hub_pts_for_metrics, ticks_per_day=TICKS_PER_DAY)

row = {
    "city": CITY_NAME,
    "city_slug": CITY_SLUG,
    "scenario": "baseline",
    "replicate": int(RUN_SEED),
    "ticks": int(RUN_TICKS),
    **struct_m, **mob_m, **dens_m, **water_m, **epi_m,
}
out_path = append_analysis_row(row)
print(f"[analysis_table] appended baseline row → {out_path}")

# ===================== Notebook cell 45 =====================
# R4b — Trip-flow Information Entropy (daily) from saved OD flows (fallback: snapshots)
# Output: shared_data/{CITY_SLUG}_entropy.csv with columns:
# day, trips_total, H_out, H_in, H_total
#
# Output files:
#   {CITY_SLUG}_tripflows.csv
#   {CITY_SLUG}_tripflows_daily.csv
# so the helper notebook has OD/trip info even for old runs.

import os, glob, numpy as np, pandas as pd, geopandas as gpd

assert 'CITY_SLUG' in globals() and 'SHARED_OUT' in globals(), "Run R0a/R0b first."
os.makedirs(SHARED_OUT, exist_ok=True)

def _find_od_path(slug):
    # include the R4 export name + a few alternates
    candidates = [
        os.path.join(SHARED_OUT, f"{slug}_tripflows.csv"),     # <-- R4 export (preferred)
        os.path.join(SHARED_OUT, f"{slug}_od_flows.csv"),
        os.path.join(SHARED_OUT, f"{slug}_od.csv"),
        os.path.join(SHARED_OUT, f"{slug}_flows.csv"),
        os.path.join(SHARED_OUT, f"{slug}_trips_od.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # last-resort: any csv with 'od' or 'tripflow' in name
    hits = glob.glob(os.path.join(SHARED_OUT, f"{slug}_*od*.csv"))
    hits += glob.glob(os.path.join(SHARED_OUT, f"{slug}_*tripflow*.csv"))
    return hits[0] if hits else None

def _standardize_od_columns(df):
    """
    Accepts either:
      - day, origin_id, dest_id, trips          (R3/R4 format)
      - day, origin_poly_id, dest_poly_id, trips
    Returns a DF with columns: day, origin_poly_id, dest_poly_id, trips
    """
    cols = set(df.columns)
    # already standard?
    if {"day", "origin_poly_id", "dest_poly_id", "trips"}.issubset(cols):
        return df[["day", "origin_poly_id", "dest_poly_id", "trips"]].copy()

    # R3/R4 naming:
    if {"day", "origin_id", "dest_id", "trips"}.issubset(cols):
        out = df.rename(columns={"origin_id": "origin_poly_id", "dest_id": "dest_poly_id"})
        return out[["day", "origin_poly_id", "dest_poly_id", "trips"]].copy()

    # generic fallback if someone wrote origin/dest
    if {"day", "origin", "dest", "trips"}.issubset(cols):
        out = df.rename(columns={"origin": "origin_poly_id", "dest": "dest_poly_id"})
        return out[["day", "origin_poly_id", "dest_poly_id", "trips"]].copy()

    # not recognized
    return None

def _entropy_from_matrix(df_day):
    """
    df_day: DataFrame with columns [origin_poly_id, dest_poly_id, trips]
    Returns weighted outbound entropy H_out, weighted inbound entropy H_in, and
    overall H_total (entropy of marginal destination distribution) for that day.
    """
    if df_day is None or df_day.empty:
        return 0.0, 0.0, 0.0, 0

    M = df_day.pivot_table(index="origin_poly_id", columns="dest_poly_id",
                           values="trips", aggfunc="sum", fill_value=0).astype(float)
    trips_total = float(M.values.sum())
    if trips_total <= 0:
        return 0.0, 0.0, 0.0, 0

    # outbound entropy per origin (weighted by origin outflow)
    row_sums = M.sum(axis=1).values
    with np.errstate(divide='ignore', invalid='ignore'):
        Prow = (M.values.T / np.where(row_sums > 0, row_sums, 1)).T

    def _H(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum()) if p.size else 0.0

    H_rows = np.array([_H(Prow[i, :]) for i in range(Prow.shape[0])])
    w_rows = row_sums / trips_total
    H_out = float(np.nansum(w_rows * H_rows))

    # inbound entropy per destination (weighted by column inflow)
    col_sums = M.sum(axis=0).values
    with np.errstate(divide='ignore', invalid='ignore'):
        Pcol = M.values / np.where(col_sums > 0, col_sums, 1)
    H_cols = np.array([_H(Pcol[:, j]) for j in range(Pcol.shape[1])])
    w_cols = col_sums / trips_total
    H_in = float(np.nansum(w_cols * H_cols))

    # overall entropy of destination marginal
    p_dest = col_sums / trips_total if trips_total > 0 else np.zeros_like(col_sums)
    H_total = _H(p_dest)

    return H_out, H_in, H_total, int(trips_total)

# 1) Try to load OD flows saved in R4
od_path = _find_od_path(CITY_SLUG)
od_df = None
if od_path:
    try:
        raw = pd.read_csv(od_path)
        od_df = _standardize_od_columns(raw)
        if od_df is None:
            print(f"[R4b] {os.path.basename(od_path)} has unrecognized columns; rebuilding from snapshots.")
    except Exception as e:
        print(f"[R4b] Failed to read {od_path}: {e}")
        od_df = None

# 2) If not available/recognized, rebuild OD counts from snapshots (fallback)
if od_df is None:
    assert 'base' in globals() and 'inner_gdf' in globals(), "Need snapshots and inner_gdf."
    snaps = base.get("snapshots", [])
    if not snaps:
        raise RuntimeError("No snapshots found; run R4 first.")
    inner = inner_gdf.to_crs("EPSG:3857")[["geometry"]].copy()
    if "poly_id" not in inner.columns:
        inner = inner.reset_index().rename(columns={"index": "poly_id"})
    inner = gpd.GeoDataFrame(inner, geometry="geometry", crs="EPSG:3857")

    rows = []
    TPD = globals().get("TICKS_PER_DAY", 24)
    for t in range(1, len(snaps)):
        day_t = (t // TPD) + 1
        g_prev = snaps[t-1][["unique_id", "current_location"]].copy()
        g_prev = gpd.GeoDataFrame(g_prev, geometry="current_location", crs="EPSG:3857")
        g_prev = gpd.sjoin(g_prev, inner, predicate="within", how="left")[["unique_id", "poly_id"]].rename(columns={"poly_id": "o"})
        g_cur = snaps[t][["unique_id", "current_location"]].copy()
        g_cur = gpd.GeoDataFrame(g_cur, geometry="current_location", crs="EPSG:3857")
        g_cur = gpd.sjoin(g_cur, inner, predicate="within", how="left")[["unique_id", "poly_id"]].rename(columns={"poly_id": "d"})
        j = g_prev.merge(g_cur, on="unique_id", how="inner")
        j = j[(j["o"].notna()) & (j["d"].notna())]
        if j.empty:
            continue
        cnt = (j.groupby(["o", "d"])
                 .size()
                 .reset_index(name="trips")
                 .rename(columns={"o": "origin_poly_id", "d": "dest_poly_id"}))
        cnt["day"] = day_t
        rows.append(cnt)

    od_df = (pd.concat(rows, ignore_index=True)
             if rows else pd.DataFrame(columns=["day", "origin_poly_id", "dest_poly_id", "trips"]))

# 3) Compute entropy per day
if od_df.empty:
    ent = pd.DataFrame({"day": [1, 2, 3, 4, 5], "trips_total": 0, "H_out": 0.0, "H_in": 0.0, "H_total": 0.0})
else:
    ent_rows = []
    for d, g in od_df.groupby("day"):
        H_out, H_in, H_total, n = _entropy_from_matrix(g[["origin_poly_id", "dest_poly_id", "trips"]])
        ent_rows.append({"day": int(d), "trips_total": n, "H_out": H_out, "H_in": H_in, "H_total": H_total})
    ent = pd.DataFrame(ent_rows).sort_values("day")

# 4) Save entropy
out_csv = os.path.join(SHARED_OUT, f"{CITY_SLUG}_entropy.csv")
ent.to_csv(OUT_DIR /out_csv, index=False)
print(f"[R4b] Wrote {out_csv}")


# 5) (NEW) ALSO save trip flows + daily totals from od_df (for helper notebook)
if od_df is not None and not od_df.empty:
    # Ensure standard columns
    od_save = od_df[["day", "origin_poly_id", "dest_poly_id", "trips"]].copy()

    # Full OD flows
    tf_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_tripflows.csv")
    od_save.to_csv(OUT_DIR /tf_path, index=False)

    # Daily totals
    tfd = (od_save.groupby("day")["trips"]
                  .sum()
                  .reset_index(name="total_trips"))
    tfd_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_tripflows_daily.csv")
    tfd.to_csv(OUT_DIR /tfd_path, index=False)

    print(f"[R4b] Also wrote trip flows:")
    print("   ", os.path.abspath(tf_path))
    print("   ", os.path.abspath(tfd_path))
else:
    print("[R4b] No OD flows in od_df → no tripflows*.csv written.")

# ===================== Notebook cell 46 =====================
# R4c — Spatial trip-attraction map (inbound trips per inner polygon)
# Works even if no *_tripflows.csv exists: uses od_df from R4b or rebuilds from snapshots.

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

assert 'CITY_SLUG' in globals() and 'SHARED_OUT' in globals(), "Run R0a/R0b first."
assert 'inner_gdf' in globals() and inner_gdf is not None and not inner_gdf.empty, \
    "inner_gdf missing or empty (run Voronoi/inner-cell step first)."

# ─── 1. Get OD flows: prefer od_df from R4b; else rebuild from snapshots ───
od_local = None

# Case A: R4b already ran → od_df exists
if 'od_df' in globals() and isinstance(od_df, pd.DataFrame) and not od_df.empty:
    od_local = od_df.copy()
    print(f"[R4c] Using in-memory od_df from R4b for {CITY_SLUG} (rows={len(od_local)})")

# Case B: try to load from disk using R4b helpers (if available)
if od_local is None and '_find_od_path' in globals():
    path = _find_od_path(CITY_SLUG)
    if path and os.path.exists(path):
        raw = pd.read_csv(path)
        if '_standardize_od_columns' in globals():
            od_local = _standardize_od_columns(raw)
        else:
            cols = set(raw.columns)
            if {"day","origin_id","dest_id","trips"}.issubset(cols):
                od_local = raw.rename(columns={
                    "origin_id": "origin_poly_id",
                    "dest_id":   "dest_poly_id"
                })[["day","origin_poly_id","dest_poly_id","trips"]]
        if od_local is not None:
            print(f"[R4c] Loaded OD flows from file: {os.path.basename(path)} (rows={len(od_local)})")

# Case C: rebuild OD flows from snapshots (no reliance on *_tripflows.csv)
if od_local is None:
    print("[R4c] No OD CSV / od_df – rebuilding from snapshots (like R4b fallback).")
    assert 'base' in globals(), "Need `base` result from R4. Run R4 first."
    snaps = base.get("snapshots", [])
    if not snaps:
        raise RuntimeError("No snapshots in `base`; rerun R4 with save_snapshots=True.")

    inner = inner_gdf.to_crs("EPSG:3857")[["geometry"]].copy()
    if "poly_id" not in inner.columns:
        inner = inner.reset_index().rename(columns={"index": "poly_id"})
    inner = gpd.GeoDataFrame(inner, geometry="geometry", crs="EPSG:3857")

    rows = []
    TPD = globals().get("TICKS_PER_DAY", 24)
    for t in range(1, len(snaps)):
        day_t = (t // TPD) + 1

        # previous tick
        g_prev = snaps[t-1][["unique_id","current_location"]].copy()
        g_prev = gpd.GeoDataFrame(g_prev, geometry="current_location", crs="EPSG:3857")
        g_prev = gpd.sjoin(
            g_prev, inner, predicate="within", how="left"
        )[["unique_id","poly_id"]].rename(columns={"poly_id":"o"})

        # current tick
        g_cur = snaps[t][["unique_id","current_location"]].copy()
        g_cur = gpd.GeoDataFrame(g_cur, geometry="current_location", crs="EPSG:3857")
        g_cur = gpd.sjoin(
            g_cur, inner, predicate="within", how="left"
        )[["unique_id","poly_id"]].rename(columns={"poly_id":"d"})

        j = g_prev.merge(g_cur, on="unique_id", how="inner")
        j = j[(j["o"].notna()) & (j["d"].notna())]
        if j.empty:
            continue

        cnt = (
            j.groupby(["o","d"])
             .size()
             .reset_index(name="trips")
             .rename(columns={"o":"origin_poly_id","d":"dest_poly_id"})
        )
        cnt["day"] = day_t
        rows.append(cnt)

    if rows:
        od_local = pd.concat(rows, ignore_index=True)
    else:
        od_local = pd.DataFrame(columns=["day","origin_poly_id","dest_poly_id","trips"])

    print(f"[R4c] Rebuilt OD flows from snapshots (rows={len(od_local)})")

# Final guard
if od_local is None or od_local.empty:
    raise RuntimeError("No OD flows available: od_df / OD CSV / snapshots reconstruction all empty.")

# Ensure standard columns
needed_cols = {"day","origin_poly_id","dest_poly_id","trips"}
if not needed_cols.issubset(set(od_local.columns)):
    raise ValueError(f"OD columns not standard: got {od_local.columns.tolist()}")

# ─── 2. Aggregate inbound trips per destination polygon ───────────────────────
inbound = (
    od_local
    .groupby("dest_poly_id")["trips"]
    .sum()
    .reset_index(name="inbound_trips")
)

# ─── 3. Attach inbound totals to inner_gdf ────────────────────────────────────
inner_plot = inner_gdf.copy()
if "poly_id" not in inner_plot.columns:
    inner_plot = inner_plot.reset_index().rename(columns={"index": "poly_id"})

inner_plot = inner_plot.merge(
    inbound,
    left_on="poly_id",
    right_on="dest_poly_id",
    how="left"
)
inner_plot["inbound_trips"] = inner_plot["inbound_trips"].fillna(0)

# ─── 4. Harmonise CRS & supporting layers ─────────────────────────────────────
target_crs = "EPSG:3857"
inner_plot = inner_plot.to_crs(target_crs)

outer_plot = None
if "outer_gdf" in globals() and outer_gdf is not None and not outer_gdf.empty:
    outer_plot = outer_gdf.to_crs(target_crs)

boundary_plot = None
if "R_GLOBAL_M" in globals() and "boundaries_by_radius" in globals():
    R_key = int(R_GLOBAL_M)
    if R_key in boundaries_by_radius:
        boundary_plot = boundaries_by_radius[R_key].to_crs(target_crs)

# ─── 5. Plot: inbound trips choropleth + outer/Voronoi + R_GLOBAL + basemap ──
fig, ax = plt.subplots(figsize=(10, 10))

# Inbound trips per inner polygon
inner_plot.plot(
    column="inbound_trips",
    ax=ax,
    cmap="YlOrRd",
    linewidth=0.4,
    edgecolor="gray",
    legend=True,
    legend_kwds={
        "label": "Total inbound trips (all days)",
        "shrink": 0.6
    },
    alpha=0.9,
)

# Outer / Voronoi cells
if outer_plot is not None and not outer_plot.empty:
    outer_plot.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.0,
        alpha=0.6,
        label="Outer/Voronoi cells"
    )

# R_GLOBAL boundary
if boundary_plot is not None and not boundary_plot.empty:
    boundary_plot.boundary.plot(
        ax=ax,
        color="blue",
        linewidth=1.6,
        linestyle="--",
        label=f"R_GLOBAL ≈ {float(R_GLOBAL_M)/1000:.1f} km"
    )

# Basemap
try:
    ctx.add_basemap(ax, crs=target_crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.6)
except Exception as e:
    print("[basemap] skipped:", e)

# Extent: prefer R_GLOBAL boundary, else inner polygons
if boundary_plot is not None and not boundary_plot.empty:
    minx, miny, maxx, maxy = boundary_plot.total_bounds
else:
    minx, miny, maxx, maxy = inner_plot.total_bounds

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_axis_off()
ax.set_title(f"Trip attraction (inbound flows) by inner polygon — {CITY_SLUG}", fontsize=13)

handles, labels = ax.get_legend_handles_labels()
if labels:
    uniq = {}
    for h, l in zip(handles, labels):
        if l and l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", frameon=True, fontsize=8)

plt.tight_layout()
# ─── SAVE FIGURE (similar style to R5) ──────────────────────────────
# Prefer CITY_OUT if defined (per-city figs), else fall back to SHARED_OUT
out_dir = globals().get("CITY_OUT", SHARED_OUT)
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, f"{CITY_SLUG}_trip_attraction_map.png")
plt.savefig(OUT_DIR /out_png, dpi=300, bbox_inches="tight")
#plt.show()
#print(f"[Fig.trip] Saved {out_png}")

# ===================== Notebook cell 47 =====================
# R5a — BUILD inner_density (FACILITY density per inner polygon, points/km²)

import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point

def _to3857(gdf):
    if gdf is None: 
        return None
    if str(getattr(gdf, "crs", None)) == "EPSG:3857":
        return gdf
    try:
        return gdf.to_crs("EPSG:3857")
    except Exception:
        return gdf

# 0) Base polygons
if 'inner_gdf' not in globals() or inner_gdf is None or inner_gdf.empty:
    raise RuntimeError("inner_gdf is required to compute inner_density.")
inner_poly = _to3857(inner_gdf.copy())
if 'poly_id' not in inner_poly.columns:
    inner_poly['poly_id'] = np.arange(len(inner_poly))

# 1) Assemble ONLY synthetic facility POIs (taps, toilets, schools, markets)
facility_gdfs = []

# a) Drinking water taps: prefer dwater_gdf; else build from drinking_points if present
if 'dwater_gdf' in globals() and dwater_gdf is not None and not dwater_gdf.empty:
    facility_gdfs.append(_to3857(dwater_gdf[['geometry']].assign(src='taps')))
elif 'drinking_points' in globals() and drinking_points:
    taps_df = gpd.GeoDataFrame({'src':'taps'}, geometry=list(drinking_points), crs=inner_poly.crs)
    facility_gdfs.append(_to3857(taps_df))

# b) Toilets
if 'toilets_gdf' in globals() and toilets_gdf is not None and not toilets_gdf.empty:
    facility_gdfs.append(_to3857(toilets_gdf[['geometry']].assign(src='toilets')))

# c) Schools
if 'schools_gdf' in globals() and schools_gdf is not None and not schools_gdf.empty:
    facility_gdfs.append(_to3857(schools_gdf[['geometry']].assign(src='schools')))

# d) Markets
if 'markets_gdf' in globals() and markets_gdf is not None and not markets_gdf.empty:
    facility_gdfs.append(_to3857(markets_gdf[['geometry']].assign(src='markets')))

# 2) Merge facility POIs
if facility_gdfs:
    poi_all = gpd.GeoDataFrame(pd.concat(facility_gdfs, ignore_index=True), crs="EPSG:3857")
else:
    poi_all = gpd.GeoDataFrame({'src':[]}, geometry=[], crs="EPSG:3857")

# 3) Clip POIs to city boundary if available (safe)
if 'gdf_bound_3857' in globals() and gdf_bound_3857 is not None and not gdf_bound_3857.empty and not poi_all.empty:
    try:
        bound_union = gdf_bound_3857.union_all()
    except Exception:
        bound_union = gdf_bound_3857.unary_union
    poi_all = poi_all[poi_all.geometry.within(bound_union)]

# 4) Count POIs per inner polygon and normalize by km²
inner_density = inner_poly[['poly_id','geometry']].copy()
inner_density['area_km2'] = inner_density.geometry.area / 1e6

if not poi_all.empty:
    joined = gpd.sjoin(
        poi_all[['src','geometry']],
        inner_density[['poly_id','geometry']],
        how='left',
        predicate='within'
    )
    counts = joined.groupby('poly_id').size().rename('poi_count').reset_index()
else:
    counts = pd.DataFrame({'poly_id': inner_density['poly_id'], 'poi_count': 0})

inner_density = inner_density.merge(counts, on='poly_id', how='left').fillna({'poi_count':0})
inner_density['poi_density'] = inner_density['poi_count'] / inner_density['area_km2'].replace({0: np.nan})
inner_density['poi_density'] = inner_density['poi_density'].fillna(0.0)

total_pois = int(inner_density['poi_count'].sum())
print(f"inner_density built: {len(inner_density)} polygons, total facility POIs counted = {total_pois}")

# ===================== Notebook cell 48 =====================
# R5 — SPATIAL ROW (Initial, Peak, Final) — global linear scale, bright + dramatic
# - Exposed + infectious agents (state in {"E", "I"}) used for Gaussian infection field
# - One shared global colorbar (same normalization for all panels)
# - Light background, bright colormap, clear differences across days

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from shapely.ops import unary_union
from scipy.spatial import cKDTree

# ----------------------- knobs -----------------------
CMAP_NAME         = "turbo"  # bright + dramatic; try "viridis" or "turbo" if you prefer
GRID_SPACING      = 150.0     # meters between evaluation nodes
SIGMA_M           = 900.0     # Gaussian bandwidth (diffused E+I field)
CUTOFF_SIGMA      = 3.5       # truncate kernel at ~3.5σ
CONTOUR_LEVELS    = 14        # number of contour bands
EDGE_COLOR_OUTER  = "#111111"
EDGE_COLOR_INNER  = "#ffffff"
EDGE_W_OUTER      = 1.2
EDGE_W_INNER      = 0.9
WATER_BLUE        = "#7EAAD9"
WATER_RED         = "#e53935"

# ----------------------- helpers -----------------------
def _to3857(gdf):
    if gdf is None:
        return None
    return gdf if str(getattr(gdf, "crs", None)) == "EPSG:3857" else gdf.to_crs("EPSG:3857")

def _union(gdf):
    if gdf is None or gdf.empty:
        return None
    try:
        return gdf.union_all()
    except Exception:
        return gdf.unary_union

def _contam_lookup(geom):
    rec = None
    if 'water_contamination_by_wkb' in globals():
        try:
            rec = water_contamination_by_wkb.get(geom.wkb)
        except Exception:
            rec = None
    if rec is None and 'water_contamination' in globals():
        rec = water_contamination.get(id(geom))
    return bool(rec and rec.get("contaminated", False))

# ----------------------- inputs & layers -----------------------
assert 'base' in globals() and isinstance(base, dict) and len(base.get("snapshots", [])) > 0, \
    "Run R4 to generate `base` with snapshots first."

# City circle mask
city_circle_gdf = None
if 'boundaries_by_radius' in globals() and 'R_GLOBAL_M' in globals() and int(R_GLOBAL_M) in boundaries_by_radius:
    city_circle_gdf = _to3857(boundaries_by_radius[int(R_GLOBAL_M)])
elif 'gdf_bound_3857' in globals():
    city_circle_gdf = _to3857(gdf_bound_3857)
assert city_circle_gdf is not None and not city_circle_gdf.empty, "City boundary/circle missing."

city_circle = city_circle_gdf.geometry.iloc[0]
circle_union = _union(city_circle_gdf)

# Guides
# Prefer per-scenario (intervention) hub polygons if the scenario runner attached them.
_outer_plot_src = base.get("outer_gdf_plot", None)
_inner_plot_src = base.get("inner_gdf_plot", None)

if _outer_plot_src is not None and hasattr(_outer_plot_src, "empty") and (not _outer_plot_src.empty):
    outer_lines_raw = _to3857(_outer_plot_src)
else:
    outer_lines_raw = _to3857(outer_gdf) if 'outer_gdf' in globals() and outer_gdf is not None and not outer_gdf.empty else None

if _inner_plot_src is not None and hasattr(_inner_plot_src, "empty") and (not _inner_plot_src.empty):
    inner_lines = _to3857(_inner_plot_src)
else:
    inner_lines = _to3857(inner_gdf) if 'inner_gdf' in globals() and inner_gdf is not None and not inner_gdf.empty else None

outer_lines = _build_cell_boundaries_from_inner_outer(inner_lines, outer_lines_raw) if outer_lines_raw is not None else None



# Water
water_map = _to3857(water_gdf).copy() if 'water_gdf' in globals() else None
if water_map is not None and not water_map.empty:
    water_map["contaminated"] = [_contam_lookup(g) for g in water_map.geometry]

# ----------------------- ticks & titles -----------------------
L = len(base["snapshots"])
I_series = base["state_history"]["I"].values[:L]
t_early = 0
t_peak  = int(np.argmax(I_series))
t_late  = L - 1

def _tick_to_day(t):
    tpd = globals().get("TICKS_PER_DAY", 24)
    return int(t // tpd) + 1

day_initial = 1
day_peak    = _tick_to_day(t_peak)
day_final   = _tick_to_day(t_late)

same_day_peak_final = (day_final == day_peak)
titles = [
    f"Initial (Day {day_initial})",
    f"Peak (Day {day_peak})",
    f"Final (Day {day_final})" + (" — same as Peak" if same_day_peak_final else "")
]

# ----------------------- grid & Gaussian kernel -----------------------
minx, miny, maxx, maxy = city_circle_gdf.total_bounds
gx = np.arange(minx, maxx + GRID_SPACING, GRID_SPACING)
gy = np.arange(miny, maxy + GRID_SPACING, GRID_SPACING)
XX, YY = np.meshgrid(gx, gy)
grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

inside_mask = gpd.GeoSeries(
    gpd.points_from_xy(grid_pts[:, 0], grid_pts[:, 1]), crs="EPSG:3857"
).within(circle_union)
mask2d = inside_mask.values.reshape(YY.shape)

def _gaussian_field_for_tick(tick):
    """Return Gaussian-smoothed active infectious prevalence density field for a given tick."""
    snap = base["snapshots"][tick]
    infected = snap[snap["state"].isin(["E", "I"])]     # EXPOSED + INFECTIOUS

    Z = np.full(XX.shape, np.nan, dtype=float)
    if len(infected) == 0:
        Z[mask2d] = 0.0
        return Z

    pts = np.column_stack([
        infected["current_location"].x.values,
        infected["current_location"].y.values
    ])

    # Keep only sources within the circle (speed)
    gs = gpd.GeoSeries(gpd.points_from_xy(pts[:, 0], pts[:, 1]), crs="EPSG:3857")
    sel = gs.within(circle_union).values
    pts = pts[sel] if sel.any() else pts
    if pts.shape[0] == 0:
        Z[mask2d] = 0.0
        return Z

    tree = cKDTree(pts)
    inside_xy = grid_pts[inside_mask.values]
    rmax = CUTOFF_SIGMA * SIGMA_M
    B = 20000
    out_vals = np.zeros(inside_xy.shape[0], dtype=float)
    inv2s2 = 1.0 / (2.0 * SIGMA_M * SIGMA_M)

    for lo in range(0, inside_xy.shape[0], B):
        hi = min(lo + B, inside_xy.shape[0])
        qs = inside_xy[lo:hi]
        neigh_lists = tree.query_ball_point(qs, r=rmax)
        for j, idxs in enumerate(neigh_lists):
            if not idxs:
                continue
            neigh = pts[idxs]
            dx = neigh[:, 0] - qs[j, 0]
            dy = neigh[:, 1] - qs[j, 1]
            d2 = dx * dx + dy * dy
            out_vals[lo + j] = np.exp(-d2 * inv2s2).sum()

    Zflat = np.full(grid_pts.shape[0], np.nan, dtype=float)
    Zflat[inside_mask.values] = out_vals
    return Zflat.reshape(YY.shape)

# Build fields
Z_early = _gaussian_field_for_tick(t_early)
Z_peak  = _gaussian_field_for_tick(t_peak)
Z_late  = Z_peak.copy() if same_day_peak_final else _gaussian_field_for_tick(t_late)

# ----------------------- global linear normalization -----------------------
all_vals = np.concatenate([
    Z_early[~np.isnan(Z_early)].ravel(),
    Z_peak[~np.isnan(Z_peak)].ravel(),
    Z_late[~np.isnan(Z_late)].ravel()
])
pos_vals = all_vals[all_vals > 0]

if pos_vals.size == 0:
    vmin_global = 0.0
    vmax_global = 1.0
else:
    vmin_global = 0.0
    vmax_global = float(np.max(pos_vals))
    if not np.isfinite(vmax_global) or vmax_global <= 0:
        vmax_global = 1.0

norm_global = Normalize(vmin=vmin_global, vmax=vmax_global)
levels = np.linspace(norm_global.vmin, norm_global.vmax, CONTOUR_LEVELS)

print(f"[norm] vmin={vmin_global:.3g}, vmax={vmax_global:.3g} (linear global)")

# ----------------------- plotting -----------------------
cmap = get_cmap(CMAP_NAME)
fig, axs = plt.subplots(
    1, 3, figsize=(18, 6.5),
    gridspec_kw=dict(wspace=0.08)
)
fig.patch.set_facecolor("white")  # light background
fig.suptitle("Exposure + infectious density (states E+I): initial, peak, final", fontsize=15, fontweight="bold")

def _draw_panel(ax, Z, title):
    # Basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                        crs="EPSG:3857", zorder=0)
    except Exception:
        pass

    # Field with shared global scale
    cf = ax.contourf(
        XX, YY, Z,
        levels=levels,
        cmap=cmap,
        norm=norm_global,
        zorder=2,
        antialiased=True
    )

    # Guides
    if outer_lines is not None and not outer_lines.empty:
        outer_lines.boundary.plot(ax=ax, color=EDGE_COLOR_OUTER,
                                  linewidth=EDGE_W_OUTER, alpha=1.0, zorder=4)
    if inner_lines is not None and not inner_lines.empty:
        inner_lines.boundary.plot(ax=ax, color=EDGE_COLOR_INNER,
                                  linewidth=EDGE_W_INNER, alpha=1.0, zorder=5)

    # Water
    if water_map is not None and not water_map.empty:
        water_map.plot(
            ax=ax,
            color=water_map["contaminated"].map({False: WATER_BLUE, True: WATER_RED}),
            edgecolor="black",
            linewidth=0.35,
            alpha=0.9,
            zorder=3,
        )

    # City circle boundary
    city_circle_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=6)

    # Extent & cosmetics
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title, fontsize=13, fontweight="bold", color="black", pad=8)
    ax.axis("off")
    ax.set_facecolor("white")

    return cf

for ax, title, Z in zip(axs, titles, [Z_early, Z_peak, Z_late]):
    _draw_panel(ax, Z, title)

# Shared global colorbar on the right
sm = ScalarMappable(norm=norm_global, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs.ravel().tolist(),
                    location='right', fraction=0.03, pad=0.02)
cbar.set_label("E+I density (exposed + infectious agents)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

# Save
assert 'CITY_SLUG' in globals() and 'CITY_OUT' in globals(), "Run R0a first."
out_png = os.path.join(CITY_OUT, f"{CITY_SLUG}_fig2_spatial_row_global_bright.png")
plt.savefig(OUT_DIR /out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
#plt.show()
#print(f"[Fig.2] Saved {out_png}")

# Normalized spatial row: same ticks and KDE fields, but each panel is divided
# by its own maximum to show hotspot concentration pattern rather than absolute burden.
def _normalize_field_panel(Z):
    Zn = np.array(Z, dtype=float, copy=True)
    finite = np.isfinite(Zn)
    positive = Zn[finite & (Zn > 0)]
    if positive.size == 0:
        Zn[finite] = 0.0
        return Zn
    Zn[finite] = Zn[finite] / float(np.nanmax(positive))
    return Zn

Z_early_norm = _normalize_field_panel(Z_early)
Z_peak_norm = _normalize_field_panel(Z_peak)
Z_late_norm = _normalize_field_panel(Z_late)
norm_panel = Normalize(vmin=0.0, vmax=1.0)
levels_panel = np.linspace(0.0, 1.0, CONTOUR_LEVELS)
fig_norm, axs_norm = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw=dict(wspace=0.08))
fig_norm.patch.set_facecolor("white")
fig_norm.suptitle("Exposure + infectious density (states E+I): normalized within each panel", fontsize=15, fontweight="bold")

def _draw_panel_normalized(ax, Z, title):
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857", zorder=0)
    except Exception:
        pass
    ax.contourf(XX, YY, Z, levels=levels_panel, cmap=cmap, norm=norm_panel, zorder=2, antialiased=True)
    if outer_lines is not None and not outer_lines.empty:
        outer_lines.boundary.plot(ax=ax, color=EDGE_COLOR_OUTER, linewidth=EDGE_W_OUTER, alpha=1.0, zorder=4)
    if inner_lines is not None and not inner_lines.empty:
        inner_lines.boundary.plot(ax=ax, color=EDGE_COLOR_INNER, linewidth=EDGE_W_INNER, alpha=1.0, zorder=5)
    if water_map is not None and not water_map.empty:
        water_map.plot(ax=ax, color=water_map["contaminated"].map({False: WATER_BLUE, True: WATER_RED}), edgecolor="black", linewidth=0.35, alpha=0.9, zorder=3)
    city_circle_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=6)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_title(title, fontsize=13, fontweight="bold", color="black", pad=8)
    ax.axis("off"); ax.set_facecolor("white")

for ax, title, Z in zip(axs_norm, titles, [Z_early_norm, Z_peak_norm, Z_late_norm]):
    _draw_panel_normalized(ax, Z, title)
sm_norm = ScalarMappable(norm=norm_panel, cmap=cmap)
sm_norm.set_array([])
cbar_norm = fig_norm.colorbar(sm_norm, ax=axs_norm.ravel().tolist(), location='right', fraction=0.03, pad=0.02)
cbar_norm.set_label("Panel-normalized density (0–1)", fontsize=11)
cbar_norm.ax.tick_params(labelsize=9)
out_png_norm = os.path.join(CITY_OUT, f"{CITY_SLUG}_fig2_spatial_row_panel_normalized.png")
plt.savefig(OUT_DIR /out_png_norm, dpi=300, bbox_inches="tight", facecolor=fig_norm.get_facecolor())
plt.close(fig_norm)

# Save matching spatial row in hexbin style (same Initial / Peak / Final layout)
HEXBIN_SIZE_M = GRID_SPACING * 4.0
HEXBIN_GRIDSIZE_X = max(4, int(np.ceil((maxx - minx) / HEXBIN_SIZE_M)))
HEXBIN_EXTENT = (minx, maxx, miny, maxy)

def _hexbin_inputs_from_field(tick):
    snap = base["snapshots"][tick]
    infected = snap[snap["state"].isin(["E", "I"])]
    if len(infected) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float)
    xh = infected["current_location"].x.values.astype(float, copy=False)
    yh = infected["current_location"].y.values.astype(float, copy=False)
    gs = gpd.GeoSeries(gpd.points_from_xy(xh, yh), crs="EPSG:3857")
    sel = gs.within(circle_union).values
    if sel.any():
        xh = xh[sel]
        yh = yh[sel]
    return xh, yh, np.empty(xh.shape[0], dtype=float)

x_early_hex, y_early_hex, c_early_hex = _hexbin_inputs_from_field(t_early)
x_peak_hex,  y_peak_hex,  c_peak_hex  = _hexbin_inputs_from_field(t_peak)
x_late_hex,  y_late_hex,  c_late_hex  = _hexbin_inputs_from_field(t_late)

vmax_hex_global = 1.0
_tmp_fig, _tmp_ax = plt.subplots()
for xh, yh in zip([x_early_hex, x_peak_hex, x_late_hex], [y_early_hex, y_peak_hex, y_late_hex]):
    hb = _tmp_ax.hexbin(
        xh, yh,
        gridsize=HEXBIN_GRIDSIZE_X,
        extent=HEXBIN_EXTENT,
        mincnt=1,
    )
    if hb.get_array().size:
        _v = float(np.nanmax(hb.get_array()))
        if np.isfinite(_v) and _v > vmax_hex_global:
            vmax_hex_global = _v
    hb.remove()
plt.close(_tmp_fig)

norm_hex_global = Normalize(vmin=0.0, vmax=vmax_hex_global)

fig_hex, axs_hex = plt.subplots(
    1, 3, figsize=(18, 6.5),
    gridspec_kw=dict(wspace=0.08)
)
fig_hex.patch.set_facecolor("white")


def _draw_panel_hex(ax, xh, yh, ch, title):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    def _path_from_polygon(poly):
        verts, codes = [], []
        xy = np.asarray(poly.exterior.coords)
        verts.extend(xy.tolist())
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
        for ring in poly.interiors:
            xy = np.asarray(ring.coords)
            verts.extend(xy.tolist())
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
        return Path(np.asarray(verts, dtype=float), codes)

    def _clip_patch(ax):
        geom = city_circle_gdf.geometry.union_all() if hasattr(city_circle_gdf.geometry, "union_all") else city_circle_gdf.geometry.unary_union
        if geom.geom_type == "Polygon":
            path = _path_from_polygon(geom)
        else:
            path = Path.make_compound_path(*[_path_from_polygon(g) for g in geom.geoms if not g.is_empty])
        return PathPatch(path, transform=ax.transData)

    hb = ax.hexbin(
        xh, yh,
        gridsize=HEXBIN_GRIDSIZE_X,
        extent=HEXBIN_EXTENT,
        mincnt=1,
        cmap=cmap,
        norm=norm_hex_global,
        linewidths=0,
        edgecolors="none",
        bins="log",
        alpha=0.95,
        zorder=2,
    )
    hb.set_clip_path(_clip_patch(ax))

    if outer_lines is not None and not outer_lines.empty:
        outer_lines.boundary.plot(ax=ax, color=EDGE_COLOR_OUTER,
                                  linewidth=EDGE_W_OUTER, alpha=1.0, zorder=4)
    if inner_lines is not None and not inner_lines.empty:
        inner_lines.boundary.plot(ax=ax, color=EDGE_COLOR_INNER,
                                  linewidth=EDGE_W_INNER, alpha=1.0, zorder=5)
    if water_map is not None and not water_map.empty:
        water_map.plot(
            ax=ax,
            color=water_map["contaminated"].map({False: WATER_BLUE, True: WATER_RED}),
            edgecolor="black",
            linewidth=0.35,
            alpha=0.9,
            zorder=3,
        )
    city_circle_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=6)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title, fontsize=13, fontweight="bold", color="black", pad=8)
    ax.axis("off")
    ax.set_facecolor("white")

for ax, title, xh, yh, ch in zip(
    axs_hex, titles,
    [x_early_hex, x_peak_hex, x_late_hex],
    [y_early_hex, y_peak_hex, y_late_hex],
    [c_early_hex, c_peak_hex, c_late_hex]
):
    _draw_panel_hex(ax, xh, yh, ch, title)

sm_hex = ScalarMappable(norm=norm_hex_global, cmap=cmap)
sm_hex.set_array([])
cbar_hex = fig_hex.colorbar(sm_hex, ax=axs_hex.ravel().tolist(),
                            location='right', fraction=0.03, pad=0.02)
cbar_hex.set_label(f"Infected count per hexagon", fontsize=11)
cbar_hex.ax.tick_params(labelsize=9)

out_png_hex = os.path.join(CITY_OUT, f"{CITY_SLUG}_fig2_spatial_row_global_bright_hexbin.png")
plt.savefig(OUT_DIR /out_png_hex, dpi=300, bbox_inches="tight", facecolor=fig_hex.get_facecolor())
#plt.show()
#print(f"[Fig.2 hexbin] Saved {out_png_hex}")

# ===================== Notebook cell 49 =====================
# R5 — SPATIAL ROW (Initial, Peak, Final) — points only (no KDE)
# - Only infected ("I") plotted as points
# - Light background, clear differences across days
# - Panel titles show both Day and Tick

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# ----------------------- knobs -----------------------
EDGE_COLOR_OUTER  = "#111111"
EDGE_COLOR_INNER  = "#ffffff"
EDGE_W_OUTER      = 1.2
EDGE_W_INNER      = 0.9
WATER_BLUE        = "#7EAAD9"
WATER_RED         = "#e53935"

POINT_COLOR       = "red"    # colour of infected points
POINT_SIZE        = 6        # size of infected points
POINT_ALPHA       = 0.7      # transparency of infected points

# ----------------------- helpers -----------------------
def _to3857(gdf):
    if gdf is None:
        return None
    return gdf if str(getattr(gdf, "crs", None)) == "EPSG:3857" else gdf.to_crs("EPSG:3857")

def _union(gdf):
    if gdf is None or gdf.empty:
        return None
    try:
        return gdf.union_all()
    except Exception:
        return gdf.unary_union

def _contam_lookup(geom):
    rec = None
    if 'water_contamination_by_wkb' in globals():
        try:
            rec = water_contamination_by_wkb.get(geom.wkb)
        except Exception:
            rec = None
    if rec is None and 'water_contamination' in globals():
        rec = water_contamination.get(id(geom))
    return bool(rec and rec.get("contaminated", False))

# ----------------------- inputs & layers -----------------------
assert 'base' in globals() and isinstance(base, dict) and len(base.get("snapshots", [])) > 0, \
    "Run R4 to generate `base` with snapshots first."

# City circle mask
city_circle_gdf = None
if 'boundaries_by_radius' in globals() and 'R_GLOBAL_M' in globals() and int(R_GLOBAL_M) in boundaries_by_radius:
    city_circle_gdf = _to3857(boundaries_by_radius[int(R_GLOBAL_M)])
elif 'gdf_bound_3857' in globals():
    city_circle_gdf = _to3857(gdf_bound_3857)
assert city_circle_gdf is not None and not city_circle_gdf.empty, "City boundary/circle missing."

city_circle = city_circle_gdf.geometry.iloc[0]
circle_union = _union(city_circle_gdf)

# Guides
outer_lines = _to3857(outer_gdf) if 'outer_gdf' in globals() and outer_gdf is not None and not outer_gdf.empty else None
inner_lines = _to3857(inner_gdf) if 'inner_gdf' in globals() and inner_gdf is not None and not inner_gdf.empty else None

# Water
water_map = _to3857(water_gdf).copy() if 'water_gdf' in globals() else None
if water_map is not None and not water_map.empty:
    water_map["contaminated"] = [_contam_lookup(g) for g in water_map.geometry]

# ----------------------- ticks & titles -----------------------
L = len(base["snapshots"])
I_series = base["state_history"]["I"].values[:L]
t_early = 0
t_peak  = int(np.argmax(I_series))
t_late  = L - 1    # always use the true last tick for the Final panel

def _tick_to_day(t):
    tpd = globals().get("TICKS_PER_DAY", 24)
    return int(t // tpd) + 1

day_initial = _tick_to_day(t_early)
day_peak    = _tick_to_day(t_peak)
day_final   = _tick_to_day(t_late)

same_day_peak_final = (day_final == day_peak)

titles = [
    f"Initial (Day {day_initial}, Tick {t_early})",
    f"Peak (Day {day_peak}, Tick {t_peak})",
    f"Final (Day {day_final}, Tick {t_late})"
    + (" — same day as Peak" if same_day_peak_final else "")
]

# ----------------------- infected point helper -----------------------
def _infected_xy_for_tick(tick):
    """Return x, y arrays for infected agents at a given tick."""
    snap = base["snapshots"][tick]
    infected = snap[snap["state"].isin(["E", "I"])]
    if infected.empty:
        return np.array([]), np.array([])
    xs = infected["current_location"].x.values
    ys = infected["current_location"].y.values
    return xs, ys

# ----------------------- extent -----------------------
minx, miny, maxx, maxy = city_circle_gdf.total_bounds

# ----------------------- plotting -----------------------
fig, axs = plt.subplots(
    1, 3, figsize=(18, 6.5),
    gridspec_kw=dict(wspace=0.08)
)
fig.patch.set_facecolor("white")  # light background

def _draw_panel(ax, tick, title):
    # Basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                        crs="EPSG:3857", zorder=0)
    except Exception:
        pass

    # Infected points
    xs, ys = _infected_xy_for_tick(tick)
    if xs.size > 0:
        ax.scatter(
            xs, ys,
            s=POINT_SIZE,
            c=POINT_COLOR,
            alpha=POINT_ALPHA,
            linewidths=0.0,
            zorder=3,
        )

    # Guides
    if outer_lines is not None and not outer_lines.empty:
        outer_lines.boundary.plot(ax=ax, color=EDGE_COLOR_OUTER,
                                  linewidth=EDGE_W_OUTER, alpha=1.0, zorder=4)
    if inner_lines is not None and not inner_lines.empty:
        inner_lines.boundary.plot(ax=ax, color=EDGE_COLOR_INNER,
                                  linewidth=EDGE_W_INNER, alpha=1.0, zorder=5)

    # Water
    if water_map is not None and not water_map.empty:
        water_map.plot(
            ax=ax,
            color=water_map["contaminated"].map({False: WATER_BLUE, True: WATER_RED}),
            edgecolor="black",
            linewidth=0.35,
            alpha=0.9,
            zorder=2,
        )

    # City circle boundary
    city_circle_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=6)

    # Extent & cosmetics
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title, fontsize=13, fontweight="bold", color="black", pad=8)
    ax.axis("off")
    ax.set_facecolor("white")

# Use actual ticks, including the true final tick
ticks_for_panels = [t_early, t_peak, t_late]

for ax, title, tick in zip(axs, titles, ticks_for_panels):
    _draw_panel(ax, tick, title)

plt.tight_layout()

# ----------------------- SAVE (robust, like other figs) -----------------------
assert 'CITY_SLUG' in globals(), "CITY_SLUG is missing (run R0a first)."
# Prefer CITY_OUT if defined; else fall back to SHARED_OUT; else current dir
out_dir = globals().get("CITY_OUT", globals().get("SHARED_OUT", "."))
os.makedirs(out_dir, exist_ok=True)

out_png = os.path.join(out_dir, f"{CITY_SLUG}_fig2a_spatial_row_points.png")
plt.savefig(OUT_DIR /out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
#plt.show()
#print(f"[Fig.2a] Saved {out_png}")

# ===================== Notebook cell 50 =====================
# R5b — DIAGNOSTIC: tap & water contamination snapshot
# (CHANGED) Define `_contam_lookup` here if not already defined (so this cell works standalone).

# Helper to read contamination flags from global registries
if '_contam_lookup' not in globals():
    def _contam_lookup(geom):
        rec = None
        if 'water_contamination_by_wkb' in globals():
            try:
                rec = water_contamination_by_wkb.get(geom.wkb)
            except Exception:
                rec = None
        if rec is None and 'water_contamination' in globals():
            rec = water_contamination.get(id(geom))
        return bool(rec and rec.get("contaminated", False))

def contamination_summary(label="now"):
    n_taps = n_taps_cont = 0
    n_water = n_water_cont = 0

    # --- Natural water ---
    if 'water_gdf' in globals() and water_gdf is not None and not water_gdf.empty:
        for g in water_gdf.geometry:
            if _contam_lookup(g):
                n_water_cont += 1
            n_water += 1

    # --- Taps from both sources, dedup by WKB ---
    taps = []
    if 'dwater_gdf' in globals() and dwater_gdf is not None and not dwater_gdf.empty:
        taps.extend(list(dwater_gdf.geometry))
    if 'drinking_points' in globals() and drinking_points:
        taps.extend([pt for pt in drinking_points if pt is not None])

    seen = set(); uniq = []
    for pt in taps:
        try:
            w = pt.wkb
            if w not in seen:
                seen.add(w); uniq.append(pt)
        except Exception:
            pass

    for pt in uniq:
        if _contam_lookup(pt):
            n_taps_cont += 1
        n_taps += 1

    # --- Print summary ---
    print(f"[{label}] Natural water bodies: 0/{n_water} contaminated (spatial-only layer)")
    if n_taps > 0:
        print(f"[{label}] Taps: {n_taps_cont}/{n_taps} contaminated ({100.0*n_taps_cont/n_taps:.1f}%)")
    else:
        print(f"[{label}] Taps: none found")

# Call once before running scenarios (after R2):
contamination_summary("t=0 (after reset)")

# And once after a run:
contamination_summary("after run")

# ===================== Notebook cell 51 =====================
# R6 — NODE STATS + WEEKLY EXPOSURES (stable WKB ids; always write headers)
# (CHANGED) Use CITY_SLUG for filenames (no mono/poly tags).
# (CHANGED) Guard to ensure SHARED_OUT exists.

import os
import pandas as pd

assert 'CITY_SLUG' in globals() and 'SHARED_OUT' in globals(), "Run R0a first."
os.makedirs(SHARED_OUT, exist_ok=True)

def dump_node_stats(res, prefix):
    """
    Exports:
      - {prefix}_contam.csv : per-node contamination with stable node_id = WKB (points & polygons)
      - {prefix}_visits.csv : per-node visit counts with node_id = WKB when available (else id:XXX)
    """
    # --- CONTAMINATION: prefer WKB-keyed registry; include POINT nodes (taps) and POLYGONS (waters) ---
    rows = []
    if 'water_contamination_by_wkb' in globals() and isinstance(water_contamination_by_wkb, dict):
        for wkb, rec in water_contamination_by_wkb.items():
            geom = rec.get("geometry", None)
            if geom is None:
                continue
            try:
                gtype = getattr(geom, "geom_type", "").lower()
            except Exception:
                gtype = ""
            rows.append({
                "node_id": wkb,                                 # stable id
                "feces_load": float(rec.get("feces_load", 0.0)),
                "contaminated": int(bool(rec.get("contaminated", False))),
                "geom_type": gtype if gtype else "unknown"
            })
    contam_df = pd.DataFrame(rows, columns=["node_id","feces_load","contaminated","geom_type"])
    if contam_df.empty:
        contam_df = pd.DataFrame(columns=["node_id","feces_load","contaminated","geom_type"])
    contam_df.to_csv(os.path.join(SHARED_OUT, f"{prefix}_contam.csv"), index=False)

    # --- VISITS: map id(geom) -> WKB for taps when possible; else fall back to "id:XXX" ---
    id2wkb = {}
    if 'dwater_gdf' in globals() and dwater_gdf is not None and not dwater_gdf.empty:
        for pt in dwater_gdf.geometry:
            try:
                id2wkb[id(pt)] = pt.wkb
            except Exception:
                pass

    vis_rows = []
    node_vis = res.get("node_visits", {}) if isinstance(res, dict) else {}
    for k, v in node_vis.items():
        node_wkb = id2wkb.get(k, None)
        vis_rows.append({
            "node_id": (node_wkb if node_wkb is not None else f"id:{k}"),
            "visits": int(v)
        })
    visits_df = pd.DataFrame(vis_rows, columns=["node_id","visits"])
    if visits_df.empty:
        visits_df = pd.DataFrame(columns=["node_id","visits"])
    visits_df.to_csv(os.path.join(SHARED_OUT, f"{prefix}_visits.csv"), index=False)

def dump_weekly_exposures(res, prefix):
    """
    Exports:
      - {prefix}_weeklyexp.csv : per-agent weekly exposure counts (S->E events), week = floor(tick/24/7)
    """
    ex = res.get("exposures", pd.DataFrame())
    if ex is None or ex.empty:
        pd.DataFrame(columns=["agent","week","count"]).to_csv(
            os.path.join(SHARED_OUT, f"{prefix}_weeklyexp.csv"), index=False
        )
        return
    ex = ex.copy()
    ex["day"] = (ex["tick"] // TICKS_PER_DAY).astype(int)
    ex["week"] = (ex["day"] // 7).astype(int)
    w = ex.groupby(["agent","week"]).size().reset_index(name="count")
    w.to_csv(os.path.join(SHARED_OUT, f"{prefix}_weeklyexp.csv"), index=False)

# Run exports for baseline (CHANGED: CITY_SLUG instead of CITY_TAG)
dump_node_stats(base, CITY_SLUG)
dump_weekly_exposures(base, CITY_SLUG)

# ===================== Notebook cell 52 =====================
# R7 — Spatial intervention definitions
# Saves standard per-scenario exports and appends one row to SHARED_OUT/analysis_table.csv.

import os, json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

assert 'CITY_SLUG' in globals() and 'SHARED_OUT' in globals(), "Run R0a/R0b first."
assert 'run_scenario' in globals(), "Run R3 first."
os.makedirs(SHARED_OUT, exist_ok=True)
os.makedirs(CITY_OUT, exist_ok=True)

def _export_scenario(result, scenario_tag):
    # incidence% per tick
    trans_df = result.get("transitions", None)
    N = result.get("N", len(agent_list))
    if trans_df is None or len(trans_df) == 0:
        raise RuntimeError(f"No transitions recorded for {scenario_tag}. Make sure you re-ran R3 before R7.")

    inc_pct_tick = 100.0 * trans_df["SE"].to_numpy() / max(N, 1)
    inc_tick_series = pd.Series(inc_pct_tick, index=trans_df["tick"].to_numpy(), name="inc")

    pd.DataFrame({"tick": inc_tick_series.index, "inc": inc_tick_series.values}).to_csv(
        os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_incidence.csv"), index=False
    )

    # state_history
    if "state_history" in result and result["state_history"] is not None:
        result["state_history"].to_csv(
            os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_state_history.csv"), index=False
        )

    # FOI exports (tick-level and daily) from true S→E transitions
    foi_tick_df, foi_daily_df = build_foi_outputs(result, ticks_per_day=TICKS_PER_DAY)
    foi_tick_df.to_csv(
        os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_foi_tick.csv"), index=False
    )
    foi_daily_df.to_csv(
        os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_foi_daily.csv"), index=False
    )

    # spread JSON (if helper exists)
    if 'compute_spread_radius_km_at_peak_I' in globals():
        try:
            spread_km = compute_spread_radius_km_at_peak_I(result, center_pt)
        except Exception:
            spread_km = np.nan
        with open(os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_spread.json"), "w") as f:
            json.dump({"radius_km": spread_km}, f)

    # OD flows (if already returned)
    tf = result.get("trip_flows", None)
    tfd = result.get("trip_flows_daily", None)

    if tf is None or (hasattr(tf, "empty") and tf.empty) or tfd is None or (hasattr(tfd, "empty") and tfd.empty):
        # try rebuild if helper exists
        if '_build_od_from_snapshots' in globals():
            tf, tfd, inner_polys = _build_od_from_snapshots(result, inner_gdf)

    if tf is not None and hasattr(tf, "to_csv") and (not getattr(tf, "empty", False)):
        tf.to_csv(os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_tripflows.csv"), index=False)

    if tfd is not None and hasattr(tfd, "to_csv") and (not getattr(tfd, "empty", False)):
        tfd.to_csv(os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_tripflows_daily.csv"), index=False)

    actual_trips = result.get("actual_trips", None)
    if actual_trips is not None and hasattr(actual_trips, "to_csv"):
        actual_trips.to_csv(os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_actual_trips.csv"), index=False)

    facility_visits = result.get("facility_visits", None)
    if facility_visits is not None and hasattr(facility_visits, "to_csv"):
        facility_visits.to_csv(os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_facility_visits.csv"), index=False)

def _append_metrics(result, scenario_tag):
    trans_df = result.get("transitions", None)
    N = result.get("N", len(agent_list))

    # load trip flows from memory if present
    trip_flows_df = result.get("trip_flows", None)

    # inner polygon centroids
    inner_polys = result.get("inner_polys", None)
    if inner_polys is not None and hasattr(inner_polys, "empty") and (not inner_polys.empty):
        INNER_POLY_CENTROIDS = _ensure_inner_centroids(inner_polys)
    else:
        INNER_POLY_CENTROIDS = {}

    inner_centroids = INNER_POLY_CENTROIDS

    # hub points for attraction-based structure
    hub_pts_for_metrics = None
    if 'centroid_gdf_new' in globals() and centroid_gdf_new is not None and not centroid_gdf_new.empty:
        hub_pts_for_metrics = centroid_gdf_new[["geometry"]].copy()
        hub_pts_for_metrics = hub_pts_for_metrics.set_crs(activity_feats.crs if 'activity_feats' in globals() else "EPSG:3857")
    elif 'activity_feats' in globals() and activity_feats is not None and not activity_feats.empty:
        hub_pts_for_metrics = activity_feats[["geometry"]].copy()

    struct_m = compute_structure_metrics_from_hub_attraction(trip_flows_df, hub_pts_for_metrics)
    mob_m = compute_mobility_metrics(trip_flows_df, inner_centroids, N_agents=N, ticks=RUN_TICKS, ticks_per_day=TICKS_PER_DAY)

    # boundary for density/water metrics
    _boundary_poly = None
    if 'boundary_gdf' in globals() and boundary_gdf is not None and not boundary_gdf.empty:
        _boundary_poly = boundary_gdf.to_crs("EPSG:3857").unary_union
    elif 'gdf_bound_3857' in globals() and gdf_bound_3857 is not None and not gdf_bound_3857.empty:
        _boundary_poly = gdf_bound_3857.to_crs("EPSG:3857").unary_union

    dens_m = compute_density_metrics_from_worldpop(_boundary_poly, POP_RASTER if 'POP_RASTER' in globals() else None)
    water_m = compute_water_metrics(
        _boundary_poly,
        taps_gdf=taps_gdf if 'taps_gdf' in globals() else None,
        water_polys_gdf=water_gdf if 'water_gdf' in globals() else None
    )

    epi_m = compute_epidemic_metrics(result, inner_centroids, hub_points_for_dispersion=hub_pts_for_metrics, ticks_per_day=TICKS_PER_DAY)

    row = {
        "city": CITY_NAME,
        "city_slug": CITY_SLUG,
        "scenario": scenario_tag,
        "replicate": int(RUN_SEED),
        "ticks": int(RUN_TICKS),
        **struct_m, **mob_m, **dens_m, **water_m, **epi_m,
    }
    out_path = append_analysis_row(row)
    print(f"[analysis_table] appended {scenario_tag} row → {out_path}")

# ---- determine extra hubs (distribution-grounded; replaces "30% of baseline") ----
# We derive "how many" extra centres to add from *how concentrated* the current hub system is.
# Intuition: the more monocentric the city (one dominant hub), the more additional subcentres we add.
#
# We operationalise "monocentric vs polycentric" using a standard concentration/entropy idea:
# - Compute weights for each existing hub (proxy = number of activity points/POIs in its cluster).
# - Compute Shannon entropy H = -Σ p_i log(p_i), normalised by log(K) so it lies in [0,1]
#   where K is number of hubs. 1.0 means perfectly even (very polycentric); 0.0 means one hub dominates.
# - Define dominance = 1 - H_norm. Extra hubs scale with dominance.
#
# This is consistent with the polycentricity literature that uses entropy / concentration of activity or
# employment to describe monocentric ↔ polycentric structure, and with work showing that subcentres
# emerge as cities decentralise (e.g., Louf & Barthelemy 2013; entropy-based polycentric indicators in
# planning literature).
def _compute_extra_hubs_from_distribution():
    # existing hub count
    if 'centroid_gdf_new' in globals() and centroid_gdf_new is not None and (not centroid_gdf_new.empty):
        hubs_gdf = centroid_gdf_new
    elif 'activity_feats' in globals() and activity_feats is not None and (not activity_feats.empty):
        hubs_gdf = activity_feats
    else:
        return 0, 0, 0.0, 0.0  # (n_base, extra, H_norm, dominance)

    n_base = int(len(hubs_gdf))
    if n_base <= 1:
        return n_base, 1, 0.0, 1.0  # with 0/1 hubs, "dominance" is effectively maximal

    # --- weights per hub (cluster size if available; else uniform) ---
    weights = None
    if 'activity_feats' in globals() and activity_feats is not None and (not activity_feats.empty) and ('cluster' in activity_feats.columns):
        # count POIs per cluster as a proxy of hub "attractiveness"/size
        vc = activity_feats['cluster'].value_counts().sort_index()
        weights = vc.to_numpy(dtype=float)
    elif 'centroid_gdf_new' in globals() and centroid_gdf_new is not None and (not centroid_gdf_new.empty) and ('cluster' in centroid_gdf_new.columns):
        vc = centroid_gdf_new['cluster'].value_counts().sort_index()
        weights = vc.to_numpy(dtype=float)

    if weights is None or len(weights) == 0:
        weights = np.ones(n_base, dtype=float)

    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights)

    p = weights / weights.sum()
    # Shannon entropy (natural log); guard against log(0)
    H = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
    H_norm = float(H / np.log(len(p))) if len(p) > 1 else 0.0
    H_norm = float(np.clip(H_norm, 0.0, 1.0))

    dominance = float(1.0 - H_norm)

    # Extra hubs proportional to dominance (more monocentric => more added centres)
    # Scale is dimensionless and tied to the existing system; we cap to keep it realistic.
    extra = int(np.ceil(dominance * n_base))
    extra = int(np.clip(extra, 1, max(1, int(np.ceil(0.5 * n_base)))))  # cap at 50% of base hubs

    return n_base, extra, H_norm, dominance



def _peak_daily_incidence_percent_from_result(result, ticks_per_day=24):
    trans_df = result.get("transitions", None)
    if trans_df is None or len(trans_df) == 0:
        return np.nan, pd.DataFrame(columns=["day", "daily_incidence_pct"])
    N = int(result.get("N", max(1, len(agent_list))))
    inc_pct_tick = 100.0 * pd.to_numeric(trans_df["SE"], errors="coerce").fillna(0.0).to_numpy() / max(N, 1)
    day = (pd.to_numeric(trans_df["tick"], errors="coerce").fillna(0).astype(int) // int(ticks_per_day)) + 1
    daily_df = pd.DataFrame({"day": day, "daily_incidence_pct": inc_pct_tick}).groupby("day", as_index=False)["daily_incidence_pct"].sum()
    peak_val = float(daily_df["daily_incidence_pct"].max()) if not daily_df.empty else np.nan
    return peak_val, daily_df


def _save_sensitivity_heatmap_facet(df, metric_col, value_label, out_png, cmap="cividis"):
    if df is None or df.empty:
        return
    if {'k_geom', 'delta_dest', 'delta_fac'}.issubset(df.columns):
        panel_col = 'k_geom'
        y_col = 'delta_fac'
        x_col = 'delta_dest'
        panel_label = r"$k_{\mathrm{geom}}$"
        y_label = r"$\delta_{\mathrm{fac}}$"
        x_label = r"$\delta_{\mathrm{dest}}$"
    else:
        panel_col = 'lambda'
        y_col = 'eta'
        x_col = 'beta'
        panel_label = r"$\lambda$"
        y_label = r"$\eta$"
        x_label = r"$\beta$"

    panel_vals = sorted(pd.unique(df[panel_col]))
    y_vals = sorted(pd.unique(df[y_col]))
    x_vals = sorted(pd.unique(df[x_col]))
    if not panel_vals or not y_vals or not x_vals:
        return

    metric_vals = pd.to_numeric(df[metric_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(metric_vals)
    if not finite.any():
        return
    vmin = float(np.nanmin(metric_vals[finite]))
    vmax = float(np.nanmax(metric_vals[finite]))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    n_panels = len(panel_vals)
    ncols = min(5, n_panels)
    nrows = int(np.ceil(n_panels / float(ncols)))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.6 * nrows), constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    im = None
    for ax, pval in zip(axs, panel_vals):
        sub = df.loc[df[panel_col] == pval].copy()
        pivot = sub.pivot(index=y_col, columns=x_col, values=metric_col).reindex(index=y_vals, columns=x_vals)
        im = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals])
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(panel_label + " = " + f"{float(pval):.2f}")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("black")
    for ax in axs[len(panel_vals):]:
        ax.axis("off")
    if im is not None:
        cbar = fig.colorbar(im, ax=axs[:len(panel_vals)].tolist(), shrink=0.9, pad=0.02)
        cbar.set_label(value_label)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_intervention_d_sensitivity_analysis(ticks=RUN_TICKS, seed=RUN_SEED, ticks_per_day=TICKS_PER_DAY):
    k_geom_values = [0.50, 0.75, 1.00, 1.25, 1.50]
    delta_dest_values = [0.00, 0.10, 0.20, 0.30, 0.40]
    delta_fac_values = [0.00, 0.05, 0.10, 0.15, 0.20]

    lambda_values = [0.20, 0.35, 0.50, 0.65, 0.80]
    eta_values = [0.8, 1.2, 1.6, 2.0, 2.4]
    beta_values = [0.0, 0.1, 0.2, 0.35, 0.5]

    summary_rows = []
    incidence_tick_rows = []
    incidence_day_rows = []
    foi_tick_rows = []
    foi_day_rows = []

    combos = [(k_geom, delta_dest, delta_fac) for k_geom in k_geom_values for delta_dest in delta_dest_values for delta_fac in delta_fac_values]
    for k_geom, delta_dest, delta_fac in tqdm(combos, desc=f"{CITY_SLUG} D-sensitivity", total=len(combos), leave=True):
        scenario_tag = f"sensD_kgeom{k_geom:.2f}_ddest{delta_dest:.2f}_dfac{delta_fac:.2f}".replace(".", "p")
        intervention = {
            "type": "area_redistribute",
            "attractiveness_mode": "centrality",
            "eta": 1.6,
            "beta": 0.20,
            "lambda": 0.55,
            "k_geom": float(k_geom),
            "delta_dest": float(delta_dest),
            "delta_fac": float(delta_fac),
            "seed": int(seed),
            "preserve_baseline_epi": True,
        }
        result = run_scenario(ticks=ticks, seed=seed, intervention=intervention)

        peak_daily_inc, daily_inc_df = _peak_daily_incidence_percent_from_result(result, ticks_per_day=ticks_per_day)
        foi_tick_df, foi_daily_df = build_foi_outputs(result, ticks_per_day=ticks_per_day)

        foi_peak = np.nan
        foi_mean = np.nan
        if foi_tick_df is not None and not foi_tick_df.empty and "foi" in foi_tick_df.columns:
            _foi_vals = pd.to_numeric(foi_tick_df["foi"], errors="coerce")
            foi_peak = float(np.nanmax(_foi_vals.to_numpy(dtype=float))) if np.isfinite(_foi_vals.to_numpy(dtype=float)).any() else np.nan
            foi_mean = float(np.nanmean(_foi_vals.to_numpy(dtype=float))) if np.isfinite(_foi_vals.to_numpy(dtype=float)).any() else np.nan

        summary_rows.append({
            "city": CITY_NAME,
            "city_slug": CITY_SLUG,
            "scenario": "intervD_sensitivity",
            "scenario_tag": scenario_tag,
            "k_geom": float(k_geom),
            "delta_dest": float(delta_dest),
            "delta_fac": float(delta_fac),
            "peak_daily_incidence_pct": float(peak_daily_inc) if np.isfinite(peak_daily_inc) else np.nan,
            "foi_peak": float(foi_peak) if np.isfinite(foi_peak) else np.nan,
            "foi_mean": float(foi_mean) if np.isfinite(foi_mean) else np.nan,
        })

        trans_df = result.get("transitions", None)
        if trans_df is not None and len(trans_df) > 0:
            N = int(result.get("N", max(1, len(agent_list))))
            inc_pct_tick = 100.0 * pd.to_numeric(trans_df["SE"], errors="coerce").fillna(0.0).to_numpy() / max(N, 1)
            tick_df = pd.DataFrame({
                "tick": pd.to_numeric(trans_df["tick"], errors="coerce").fillna(0).astype(int),
                "daily_incidence_pct_tick": inc_pct_tick,
            })
            tick_df.insert(0, "delta_fac", float(delta_fac))
            tick_df.insert(0, "delta_dest", float(delta_dest))
            tick_df.insert(0, "k_geom", float(k_geom))
            tick_df.insert(0, "scenario_tag", scenario_tag)
            tick_df.insert(0, "city_slug", CITY_SLUG)
            incidence_tick_rows.append(tick_df)

        if daily_inc_df is not None and not daily_inc_df.empty:
            tmp = daily_inc_df.copy()
            tmp.insert(0, "delta_fac", float(delta_fac))
            tmp.insert(0, "delta_dest", float(delta_dest))
            tmp.insert(0, "k_geom", float(k_geom))
            tmp.insert(0, "scenario_tag", scenario_tag)
            tmp.insert(0, "city_slug", CITY_SLUG)
            incidence_day_rows.append(tmp)

        if foi_tick_df is not None and not foi_tick_df.empty:
            tmp = foi_tick_df.copy()
            tmp.insert(0, "delta_fac", float(delta_fac))
            tmp.insert(0, "delta_dest", float(delta_dest))
            tmp.insert(0, "k_geom", float(k_geom))
            tmp.insert(0, "scenario_tag", scenario_tag)
            tmp.insert(0, "city_slug", CITY_SLUG)
            foi_tick_rows.append(tmp)

        if foi_daily_df is not None and not foi_daily_df.empty:
            tmp = foi_daily_df.copy()
            tmp.insert(0, "delta_fac", float(delta_fac))
            tmp.insert(0, "delta_dest", float(delta_dest))
            tmp.insert(0, "k_geom", float(k_geom))
            tmp.insert(0, "scenario_tag", scenario_tag)
            tmp.insert(0, "city_slug", CITY_SLUG)
            foi_day_rows.append(tmp)

    summary_df = pd.DataFrame(summary_rows)
    incidence_tick_long_df = pd.concat(incidence_tick_rows, ignore_index=True) if incidence_tick_rows else pd.DataFrame()
    incidence_day_long_df = pd.concat(incidence_day_rows, ignore_index=True) if incidence_day_rows else pd.DataFrame()
    foi_tick_long_df = pd.concat(foi_tick_rows, ignore_index=True) if foi_tick_rows else pd.DataFrame()
    foi_day_long_df = pd.concat(foi_day_rows, ignore_index=True) if foi_day_rows else pd.DataFrame()

    summary_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_intervD_sensitivity_summary.csv")
    inc_tick_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_intervD_sensitivity_incidence_tick_long.csv")
    inc_day_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_intervD_sensitivity_incidence_day_long.csv")
    foi_tick_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_intervD_sensitivity_foi_tick_long.csv")
    foi_day_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_intervD_sensitivity_foi_day_long.csv")

    summary_df.to_csv(summary_path, index=False)
    incidence_tick_long_df.to_csv(inc_tick_path, index=False)
    incidence_day_long_df.to_csv(inc_day_path, index=False)
    foi_tick_long_df.to_csv(foi_tick_path, index=False)
    foi_day_long_df.to_csv(foi_day_path, index=False)

    _save_sensitivity_heatmap_facet(
        summary_df,
        metric_col="peak_daily_incidence_pct",
        value_label="Peak daily incidence (%)",
        out_png=os.path.join(CITY_OUT, f"fig_sensitivity_{CITY_SLUG}_intervD_peak_daily_incidence_heatmap.png"),
        cmap="cividis",
    )
    _save_sensitivity_heatmap_facet(
        summary_df,
        metric_col="foi_peak",
        value_label="Peak FOI",
        out_png=os.path.join(CITY_OUT, f"fig_sensitivity_{CITY_SLUG}_intervD_peak_foi_heatmap.png"),
        cmap="cividis",
    )

    print(f"[Sensitivity D] saved summary → {summary_path}")
    print(f"[Sensitivity D] saved incidence tick long → {inc_tick_path}")
    print(f"[Sensitivity D] saved incidence day long → {inc_day_path}")
    print(f"[Sensitivity D] saved FOI tick long → {foi_tick_path}")
    print(f"[Sensitivity D] saved FOI day long → {foi_day_path}")

    return {
        "summary": summary_df,
        "incidence_tick_long": incidence_tick_long_df,
        "incidence_day_long": incidence_day_long_df,
        "foi_tick_long": foi_tick_long_df,
        "foi_day_long": foi_day_long_df,
    }

# ---- run interventions with OUTER progress bar ------------------------------
# Single strong-test D parameter set before running the full 125-run sensitivity analysis.
# Chosen to approximate a strong, policy-plausible decentralisation:
# - stronger inner-hub deconcentration
# - moderate-strong destination redirection
# - moderate facility reallocation
# One explicit active D parameter set.  These values are not selected by Sobol
# and are kept unchanged here; optional structure-only screening can be used to
# justify or revise them without running epidemic simulations.
ACTIVE_D_PARAMS = {
    'k_geom': 2.40,
    'delta_dest': 0.80,
    'delta_fac': 0.80,
    'lambda_blend': 0.45,
}
TEST_D_K_GEOM = ACTIVE_D_PARAMS['k_geom']          # analysis alias
TEST_D_DELTA_DEST = ACTIVE_D_PARAMS['delta_dest']  # analysis alias
TEST_D_DELTA_FAC = ACTIVE_D_PARAMS['delta_fac']    # analysis alias
RUN_D_SENSITIVITY = False

# Optional Sobol sensitivity analysis.
# Default is off so the normal baseline/B/D run remains fast and keeps all old
# plots/CSVs. Turn on from the shell with:
#   RUN_SOBOL=1 SOBOL_BASE_N=64 python code/model.py --city_slug douala --run_id 0
RUN_SOBOL = bool(int(os.environ.get("RUN_SOBOL", "0")))

# Main decentralisation/co-occurrence plotting mode. This is independent of
# Sobol and uses repeated stochastic ABM runs to estimate mean ± SD.
RUN_COOCCURRENCE_ENSEMBLE = bool(int(os.environ.get("RUN_COOCCURRENCE_ENSEMBLE", "1")))
COOCCURRENCE_N_SEEDS = int(os.environ.get("COOCCURRENCE_N_SEEDS", "3"))
COOCCURRENCE_SEED_START = int(os.environ.get("COOCCURRENCE_SEED_START", str(RUN_SEED)))
KEEP_LEGACY_PLOTS = bool(int(os.environ.get("KEEP_LEGACY_PLOTS", "1")))
RUN_MOBILITY_COMPARATOR_MAIN = bool(int(os.environ.get("RUN_MOBILITY_COMPARATOR_MAIN", "0")))
RUN_PARAMETER_SCREENING = bool(int(os.environ.get("RUN_PARAMETER_SCREENING", "0")))
COOCCURRENCE_RESUME = bool(int(os.environ.get("COOCCURRENCE_RESUME", "1")))

# Paper-figure output controls. These do not change the ABM or scenario logic;
# they only control which plots are saved after the ensemble tables are built.
RUN_PAPER_PLOTS = bool(int(os.environ.get("RUN_PAPER_PLOTS", "1")))
RUN_DIAGNOSTIC_PLOTS = bool(int(os.environ.get("RUN_DIAGNOSTIC_PLOTS", "1")))
MIN_COHORT_N_FOR_MAIN_PLOT = int(os.environ.get("MIN_COHORT_N_FOR_MAIN_PLOT", "30"))

# Parallel ensemble controls. These change execution and output handling only;
# they do not change the ABM, intervention, layout, or metric calculations.
#
# COOCCURRENCE_WORKER_ONLY=1
#     Run this process's assigned seed block, save seed-level checkpoints/events,
#     and stop before any shared summaries or figures are written.
#
# COOCCURRENCE_AGGREGATE_ONLY=1
#     Require all requested checkpoints to exist, load them without rerunning any
#     seed, then create the all-seed CSV summaries, retained plots, and clean
#     final-paper figures.
COOCCURRENCE_WORKER_ONLY = bool(int(os.environ.get("COOCCURRENCE_WORKER_ONLY", "0")))
COOCCURRENCE_AGGREGATE_ONLY = bool(int(os.environ.get("COOCCURRENCE_AGGREGATE_ONLY", "0")))
if COOCCURRENCE_WORKER_ONLY and COOCCURRENCE_AGGREGATE_ONLY:
    raise ValueError("COOCCURRENCE_WORKER_ONLY and COOCCURRENCE_AGGREGATE_ONLY cannot both be 1.")

# Worker tasks must never write partial shared summaries or partial final figures.
# The aggregate-only job is the single writer for these outputs.
if COOCCURRENCE_WORKER_ONLY:
    RUN_PAPER_PLOTS = False
    RUN_DIAGNOSTIC_PLOTS = False


# Plot/ensemble mode must never accidentally launch the expensive Sobol loop.
# Sobol can be re-enabled later by running with RUN_COOCCURRENCE_ENSEMBLE=0
# and RUN_SOBOL=1.
if RUN_COOCCURRENCE_ENSEMBLE:
    RUN_SOBOL = False

RUN_SOBOL_MOBILITY = bool(int(os.environ.get("RUN_SOBOL_MOBILITY", "1")))
RUN_SOBOL_DECENTRALISATION = bool(int(os.environ.get("RUN_SOBOL_DECENTRALISATION", "1")))
RUN_SOBOL_FULL = bool(int(os.environ.get("RUN_SOBOL_FULL", "1")))
RUN_SOBOL_DECENTRALISATION_EXTENDED = bool(int(os.environ.get("RUN_SOBOL_DECENTRALISATION_EXTENDED", "0")))
SOBOL_BASE_N = int(os.environ.get("SOBOL_BASE_N", "8"))

# Sobol stochastic repetitions.
# Default is ONE seed for a quick first run.
# To switch back without editing code, use:
#   SOBOL_DEFAULT_N_SEEDS=3
# or explicitly:
#   SOBOL_REPLICATE_SEEDS=42,43,44
# For five seeds, use:
#   SOBOL_DEFAULT_N_SEEDS=5
# or:
#   SOBOL_REPLICATE_SEEDS=42,43,44,45,46
SOBOL_DEFAULT_N_SEEDS = int(os.environ.get("SOBOL_DEFAULT_N_SEEDS", "1"))
_SOBOL_DEFAULT_REPLICATE_SEEDS = ",".join(
    str(int(RUN_SEED) + i) for i in range(max(1, int(SOBOL_DEFAULT_N_SEEDS)))
)
SOBOL_REPLICATE_SEEDS = [
    int(str(x).strip())
    for x in os.environ.get("SOBOL_REPLICATE_SEEDS", _SOBOL_DEFAULT_REPLICATE_SEEDS).split(",")
    if str(x).strip()
]
SOBOL_CALC_SECOND_ORDER = bool(int(os.environ.get("SOBOL_CALC_SECOND_ORDER", "0")))
print(f"[Sobol config] RUN_SOBOL={RUN_SOBOL} | N={SOBOL_BASE_N} | mobility={RUN_SOBOL_MOBILITY} | decentralisation={RUN_SOBOL_DECENTRALISATION} | full={RUN_SOBOL_FULL} | extended={RUN_SOBOL_DECENTRALISATION_EXTENDED}")

def _base_intervention_d_dict(mode, scenario_tag):
    """Shared fixed Intervention D settings for the three D variants.

    Active planning parameters only:
      k_geom      = strength of hub-area equalisation
      delta_dest  = strength of activity-supply redistribution
      delta_fac   = strength of tap/toilet redistribution
      lambda_blend= baseline-vs-centrality blend for generic activity mass
    """
    return {
        "type": "area_redistribute",
        "scenario_tag": scenario_tag,
        "decentralisation_mode": mode,
        "k_geom": ACTIVE_D_PARAMS['k_geom'],
        "delta_dest": ACTIVE_D_PARAMS['delta_dest'],
        "delta_fac": ACTIVE_D_PARAMS['delta_fac'],
        "lambda_blend": ACTIVE_D_PARAMS['lambda_blend'],
        "use_secondary_score_ranking": False,
        # Fixed layout seed: the 50 ensemble seeds represent ABM stochasticity,
        # not a different synthetic urban plan in every repetition.
        "seed": RUN_SEED,
        "layout_seed": RUN_SEED,
        "preserve_baseline_epi": True,
    }

scenario_runs = [
    ("intervD_destinations_only", _base_intervention_d_dict("destinations_only", "intervD_destinations_only")),
    ("intervD_facilities_only", _base_intervention_d_dict("sanitation_only", "intervD_facilities_only")),
    ("intervD_area_redistribute", _base_intervention_d_dict("all", "intervD_area_redistribute")),
]
if RUN_MOBILITY_COMPARATOR_MAIN:
    scenario_runs.insert(0, (
        "intervB_mobilityreduce",
        {"type": "mobility_reduce", "rho_mean": 0.05, "gamma_mean": 1.15,
         "max_radius_m": 2500, "seed": RUN_SEED, "preserve_baseline_epi": True},
    ))

res_B = None
res_D_dest = None
res_D_fac = None
res_D = None
res_3 = None
res_4 = None

for scenario_tag, intervention in tqdm(scenario_runs, desc=f"{CITY_SLUG} interventions", total=len(scenario_runs), leave=True):
    result = run_scenario(ticks=RUN_TICKS, seed=RUN_SEED, intervention=intervention)

    _export_scenario(result, scenario_tag)
    _append_metrics(result, scenario_tag)

    if scenario_tag == "intervB_mobilityreduce":
        res_B = result
    if scenario_tag == "intervD_destinations_only":
        res_D_dest = result
    if scenario_tag == "intervD_facilities_only":
        res_D_fac = result
    if scenario_tag == "intervD_area_redistribute":
        res_D = result

print("[DONE] Intervention exports + analysis_table rows complete.")

# ---- Additional non-invasive intervention summary + Sobol helpers ------------
def _extract_core_outcomes(result, bin_ticks=6):
    """Return compact outcome metrics from one run result.

    Used by the new summary figure and by optional Sobol analysis. It does not
    change any existing exports.
    """
    trans = result.get("transitions", pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
    state = result.get("state_history", pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
    N = max(int(result.get("N", len(agent_list))) if isinstance(result, dict) else len(agent_list), 1)

    cumulative_exposure_attack_rate_pct = np.nan
    cumulative_exposure_48h_count = np.nan
    cumulative_exposure_48h_pct = np.nan
    cumulative_exposure_168h_count = np.nan
    cumulative_exposure_168h_pct = np.nan
    peak_6hr_incidence_pct = np.nan
    if trans is not None and not getattr(trans, "empty", True) and "SE" in trans.columns:
        _se = pd.to_numeric(trans["SE"], errors="coerce").fillna(0.0)
        _ticks = pd.to_numeric(trans.get("tick", pd.Series(np.arange(len(trans)), index=trans.index)), errors="coerce")
        # Ticks are zero based in run_scenario(): tick 0 is the first simulated
        # hour. Therefore the first 48 h are ticks 0..47, and the full seven-day
        # horizon is ticks 0..167. These endpoint definitions are explicit to
        # avoid an off-by-one ambiguity in endpoint summaries.
        cumulative_exposure_48h_count = float(_se.loc[_ticks.lt(48)].sum())
        cumulative_exposure_48h_pct = 100.0 * cumulative_exposure_48h_count / float(N)
        cumulative_exposure_168h_count = float(_se.loc[_ticks.lt(168)].sum())
        cumulative_exposure_168h_pct = 100.0 * cumulative_exposure_168h_count / float(N)
        cumulative_exposure_attack_rate_pct = cumulative_exposure_168h_pct
        inc6 = six_hour_incidence_percent_from_transitions(trans, bin_ticks=bin_ticks)
        if inc6 is not None and not inc6.empty and "incidence_pct" in inc6.columns:
            peak_6hr_incidence_pct = float(pd.to_numeric(inc6["incidence_pct"], errors="coerce").max())

    foi_tick = result.get("foi_tick", pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
    if foi_tick is None or getattr(foi_tick, "empty", True) or "foi" not in foi_tick.columns:
        foi_peak = np.nan
        foi_mean = np.nan
    else:
        foi_vals = pd.to_numeric(foi_tick["foi"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        foi_peak = float(foi_vals.max()) if len(foi_vals) else np.nan
        foi_mean = float(foi_vals.mean()) if len(foi_vals) else np.nan

    peak_active_infectious_pct = np.nan
    if state is not None and not getattr(state, "empty", True):
        active6 = six_hour_active_infectious_percent(state, bin_ticks=bin_ticks)
        if active6 is not None and not active6.empty and "active_infectious_pct" in active6.columns:
            peak_active_infectious_pct = float(pd.to_numeric(active6["active_infectious_pct"], errors="coerce").max())

    return {
        "n_agents": int(N),
        "cumulative_exposure_attack_rate_pct": cumulative_exposure_attack_rate_pct,
        "cumulative_exposure_48h_count": cumulative_exposure_48h_count,
        "cumulative_exposure_48h_pct": cumulative_exposure_48h_pct,
        "cumulative_exposure_168h_count": cumulative_exposure_168h_count,
        "cumulative_exposure_168h_pct": cumulative_exposure_168h_pct,
        "peak_6hr_incidence_pct": peak_6hr_incidence_pct,
        "foi_peak": foi_peak,
        "foi_mean": foi_mean,
        "peak_active_infectious_pct": peak_active_infectious_pct,
    }


def _extract_sobol_outputs(result):
    return _extract_core_outcomes(result, bin_ticks=6)


def _save_intervention_summary_3panel(results_dict, out_dir=CITY_OUT):
    """Save a new 3-panel intervention-effect summary without replacing old plots."""
    rows = []
    for scenario, result in (results_dict or {}).items():
        if result is None:
            continue
        rows.append({"scenario": scenario, **_extract_core_outcomes(result, bin_ticks=6)})
    df = pd.DataFrame(rows)
    if df.empty:
        return None

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{CITY_SLUG}_intervention_summary_3panel.csv")
    png_path = os.path.join(out_dir, f"fig_{CITY_SLUG}_intervention_summary_3panel.png")
    pdf_path = os.path.join(out_dir, f"fig_{CITY_SLUG}_intervention_summary_3panel.pdf")
    df.to_csv(csv_path, index=False)

    labels = [_pretty_tripdiag_label(s).replace("\n", " ") if "_pretty_tripdiag_label" in globals() else s for s in df["scenario"]]
    metrics = [
        ("cumulative_exposure_attack_rate_pct", "Final cumulative exposure attack rate (%)"),
        ("peak_6hr_incidence_pct", "Peak 6-hour incidence (%)"),
        ("peak_active_infectious_pct", "Peak active infectious prevalence (%)"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (col, title) in zip(axs, metrics):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col in df.columns else np.zeros(len(df))
        ax.bar(np.arange(len(df)), vals)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel("% of agents")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Intervention summary", fontsize=13, fontweight="bold")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Intervention summary] saved → {png_path}")
    return {"summary": df, "csv": csv_path, "png": png_path, "pdf": pdf_path}

try:
    INTERVENTION_SUMMARY_3PANEL = _save_intervention_summary_3panel({
        "baseline": base,
        "intervB_mobilityreduce": res_B,
        "intervD_destinations_only": globals().get("res_D_dest", None),
        "intervD_facilities_only": globals().get("res_D_fac", None),
        "intervD_area_redistribute": res_D,
    })
except Exception as _e_summary3:
    print(f"[Intervention summary] skipped: {_e_summary3}")

# ---- Optional Sobol sensitivity analysis helpers -----------------------------
SOBOL_MOBILITY_PROBLEM = {
    "num_vars": 2,
    "names": ["rho_mean", "gamma_mean"],
    "bounds": [[0.10, 0.50], [0.60, 1.10]],
}
SOBOL_DECENTRALISATION_PROBLEM = {
    "num_vars": 4,
    "names": ["k_geom", "delta_dest", "delta_fac", "lambda_blend"],
    # k_geom is retained for compatibility but now controls area equalisation strength.
    "bounds": [[0.00, 2.50], [0.00, 0.80], [0.00, 0.80], [0.30, 0.85]],
}
SOBOL_DECENTRALISATION_ACTIVITY_PROBLEM = {
    "num_vars": 3,
    "names": ["k_geom", "delta_dest", "lambda_blend"],
    # Activity-only D: hub size + schools/markets/generic POI attraction.
    "bounds": [[0.00, 2.50], [0.00, 0.80], [0.30, 0.85]],
}
SOBOL_DECENTRALISATION_WASH_PROBLEM = {
    "num_vars": 2,
    "names": ["k_geom", "delta_fac"],
    # WASH-only D: hub size + taps/toilets equalisation.
    "bounds": [[0.00, 2.50], [0.00, 0.80]],
}
SOBOL_DECENTRALISATION_ALL_PROBLEM = SOBOL_DECENTRALISATION_PROBLEM
SOBOL_FULL_PROBLEM = {
    "num_vars": 6,
    "names": ["rho_mean", "gamma_mean", "k_geom", "delta_dest", "delta_fac", "lambda_blend"],
    "bounds": [[0.10, 0.50], [0.60, 1.10], [0.00, 2.50], [0.00, 0.80], [0.00, 0.80], [0.30, 0.85]],
}
SOBOL_DECENTRALISATION_EXTENDED_PROBLEM = {
    "num_vars": 5,
    "names": ["k_geom", "delta_dest", "delta_fac", "lambda_blend", "secondary_priority_strength"],
    "bounds": [[0.00, 2.50], [0.00, 0.80], [0.00, 0.80], [0.30, 0.85], [0.00, 1.00]],
}


def _build_mobility_sobol_intervention(params, seed):
    return {
        "type": "sobol_mobility",
        "is_sobol": True,
        "apply_mobility_params": True,
        "preserve_baseline_epi": True,
        "reset_mobility_memory": True,
        "rho_mean": float(params["rho_mean"]),
        "gamma_mean": float(params["gamma_mean"]),
        "max_radius_m": None,
        "seed": int(seed),
    }


def _build_decentralisation_activity_sobol_intervention(params, seed):
    """Sobol case D1: hub-area + activity-only decentralisation.

    Moves generic activity POIs, schools and markets, while keeping taps/toilets
    at their baseline locations.
    """
    return {
        "type": "sobol_decentralisation_activity",
        "is_sobol": True,
        "apply_area_redistribute": True,
        "preserve_baseline_epi": True,
        "reset_mobility_memory": True,
        "attractiveness_mode": "centrality",
        "eta": 2.00,
        "beta": 0.15,
        "k_geom": float(params["k_geom"]),
        "delta_dest": float(params["delta_dest"]),
        "delta_fac": 0.0,
        "lambda_blend": float(params["lambda_blend"]),
        "decentralisation_mode": "destinations_only",
        "use_secondary_score_ranking": False,
        "secondary_priority_strength": 0.50,
        "inbound_trip_share_by_cluster": globals().get("BASELINE_INBOUND_TRIP_SHARE_BY_CLUSTER", {}),
        "alpha_fac": 0.80,
        "beta_fac": 0.35,
        "seed": int(seed),
    }


def _build_decentralisation_wash_sobol_intervention(params, seed):
    """Sobol case D2: hub-area + WASH-only decentralisation.

    Moves drinking-water taps and toilets, while keeping generic POIs,
    schools and markets at their baseline locations.
    """
    return {
        "type": "sobol_decentralisation_wash",
        "is_sobol": True,
        "apply_area_redistribute": True,
        "preserve_baseline_epi": True,
        "reset_mobility_memory": True,
        "attractiveness_mode": "centrality",
        "eta": 2.00,
        "beta": 0.15,
        "k_geom": float(params["k_geom"]),
        "delta_dest": 0.0,
        "delta_fac": float(params["delta_fac"]),
        "lambda_blend": 0.45,
        "decentralisation_mode": "sanitation_only",
        "use_secondary_score_ranking": False,
        "secondary_priority_strength": 0.50,
        "inbound_trip_share_by_cluster": globals().get("BASELINE_INBOUND_TRIP_SHARE_BY_CLUSTER", {}),
        "alpha_fac": 0.80,
        "beta_fac": 0.35,
        "seed": int(seed),
    }


def _build_decentralisation_all_sobol_intervention(params, seed):
    """Sobol case D3: hub-area + all-amenity decentralisation."""
    return {
        "type": "sobol_decentralisation_all",
        "is_sobol": True,
        "apply_area_redistribute": True,
        "preserve_baseline_epi": True,
        "reset_mobility_memory": True,
        "attractiveness_mode": "centrality",
        "eta": 2.00,
        "beta": 0.15,
        "k_geom": float(params["k_geom"]),
        "delta_dest": float(params["delta_dest"]),
        "delta_fac": float(params["delta_fac"]),
        "lambda_blend": float(params["lambda_blend"]),
        "decentralisation_mode": "all",
        "use_secondary_score_ranking": False,
        "secondary_priority_strength": 0.50,
        "inbound_trip_share_by_cluster": globals().get("BASELINE_INBOUND_TRIP_SHARE_BY_CLUSTER", {}),
        "alpha_fac": 0.80,
        "beta_fac": 0.35,
        "seed": int(seed),
    }


def _build_decentralisation_sobol_intervention(params, seed):
    """Backward-compatible alias for the all-amenity D Sobol case."""
    return _build_decentralisation_all_sobol_intervention(params, seed)


def _build_decentralisation_extended_sobol_intervention(params, seed):
    d = _build_decentralisation_sobol_intervention(params, seed)
    d.update({
        "type": "sobol_decentralisation_extended",
        # Kept as optional/supplementary only. Main D now uses all hubs equally;
        # this parameter is retained for backward-compatible experiments.
        "use_secondary_score_ranking": False,
        "secondary_priority_strength": float(params["secondary_priority_strength"]),
        "inbound_trip_share_by_cluster": globals().get("BASELINE_INBOUND_TRIP_SHARE_BY_CLUSTER", {}),
    })
    return d


def _build_full_sobol_intervention(params, seed):
    d = _build_decentralisation_sobol_intervention(params, seed)
    d.update({
        "type": "sobol_full",
        "apply_mobility_params": True,
        "apply_area_redistribute": True,
        "rho_mean": float(params["rho_mean"]),
        "gamma_mean": float(params["gamma_mean"]),
    })
    return d


def _ensure_salib_for_sobol():
    """Prefer SALib, but do not crash the whole model if SALib is unavailable.

    If SALib is present, the official SALib sampler/analyser are used. If it is
    not present on the HPC environment, the script falls back to an internal
    Saltelli-style Sobol sampler/analyser using scipy.stats.qmc.Sobol. This keeps
    the run from failing after all the old plots/CSVs have been produced.
    """
    global salib_sobol_sample, salib_sobol_analyze, saltelli, _SALIB_NEW_API, _SOBOL_INTERNAL_FALLBACK
    try:
        from SALib.sample import sobol as salib_sobol_sample
        from SALib.analyze import sobol as salib_sobol_analyze
        _SALIB_NEW_API = True
        _SOBOL_INTERNAL_FALLBACK = False
        return True
    except Exception:
        try:
            from SALib.sample import saltelli
            from SALib.analyze import sobol as salib_sobol_analyze
            _SALIB_NEW_API = False
            _SOBOL_INTERNAL_FALLBACK = False
            return True
        except Exception:
            _SALIB_NEW_API = False
            _SOBOL_INTERNAL_FALLBACK = True
            print("[Sobol] SALib is not available; using internal scipy.qmc Saltelli-style fallback. Install SALib for the official SALib implementation.")
            return False


def _scale_unit_sobol_samples(unit_samples, problem):
    unit_samples = np.asarray(unit_samples, dtype=float)
    bounds = np.asarray(problem["bounds"], dtype=float)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return lo + unit_samples * (hi - lo)


def _make_internal_sobol_samples(problem, base_n, seed=RUN_SEED):
    """Internal first-order/total-order Saltelli sample matrix.

    Row order is A, B, AB_0, AB_1, ... AB_(D-1), each block with N rows.
    This gives N*(D+2) model runs when second-order terms are not requested.
    """
    from scipy.stats import qmc
    D = int(problem["num_vars"])
    N = max(2, int(base_n))
    sampler = qmc.Sobol(d=2 * D, scramble=True, seed=int(seed))
    # random_base2 is preferred for Sobol sequences when N is a power of two.
    if N > 0 and (N & (N - 1) == 0):
        unit = sampler.random_base2(m=int(np.log2(N)))
    else:
        unit = sampler.random(N)
    A = unit[:, :D]
    B = unit[:, D:]
    blocks = [_scale_unit_sobol_samples(A, problem), _scale_unit_sobol_samples(B, problem)]
    for j in range(D):
        AB = A.copy()
        AB[:, j] = B[:, j]
        blocks.append(_scale_unit_sobol_samples(AB, problem))
    return np.vstack(blocks)


def _make_sobol_samples(problem, base_n, calc_second_order=False, seed=RUN_SEED):
    has_salib = _ensure_salib_for_sobol()
    if has_salib and globals().get("_SALIB_NEW_API", False):
        globals()["_SOBOL_SAMPLE_MODE"] = "salib_new"
        globals()["_SOBOL_BASE_N_USED"] = int(base_n)
        return salib_sobol_sample.sample(problem, N=int(base_n), calc_second_order=calc_second_order, seed=int(seed))
    if has_salib:
        globals()["_SOBOL_SAMPLE_MODE"] = "salib_saltelli"
        globals()["_SOBOL_BASE_N_USED"] = int(base_n)
        return saltelli.sample(problem, int(base_n), calc_second_order=calc_second_order)
    if calc_second_order:
        print("[Sobol] internal fallback supports first-order and total-order indices only; calc_second_order=False will be used.")
    globals()["_SOBOL_SAMPLE_MODE"] = "internal"
    globals()["_SOBOL_BASE_N_USED"] = int(base_n)
    return _make_internal_sobol_samples(problem, int(base_n), seed=int(seed))


def run_sobol_experiment(label, problem, scenario_builder, base_n=128, replicate_seeds=None, ticks=RUN_TICKS):
    if replicate_seeds is None:
        replicate_seeds = [RUN_SEED]
    samples = _make_sobol_samples(problem, base_n=base_n, calc_second_order=SOBOL_CALC_SECOND_ORDER, seed=RUN_SEED)
    rows = []
    sample_rows = []
    for sample_id, x in enumerate(tqdm(samples, desc=f"{CITY_SLUG} Sobol {label}", leave=True)):
        params = {name: float(val) for name, val in zip(problem["names"], x)}
        sample_rows.append({"sample_id": int(sample_id), **params})
        for rep_id, s in enumerate(replicate_seeds):
            intervention = scenario_builder(params, seed=s)
            result = run_scenario(
                ticks=ticks,
                seed=int(s),
                intervention=intervention,
                save_snapshots=False,
                track_visits=False,
                log_exposures=False,
            )
            rows.append({
                "sobol_label": label,
                "sample_id": int(sample_id),
                "replicate_id": int(rep_id),
                "seed": int(s),
                **params,
                **_extract_sobol_outputs(result),
            })
    return samples, pd.DataFrame(sample_rows), pd.DataFrame(rows)


def _internal_sobol_analyze(problem, Y, base_n):
    """Compute S1/ST from the internal A, B, AB_i sample order."""
    D = int(problem["num_vars"])
    N = int(base_n)
    Y = np.asarray(Y, dtype=float)
    expected = N * (D + 2)
    if Y.size != expected:
        raise RuntimeError(f"Internal Sobol expected {expected} outputs for N={N}, D={D}; got {Y.size}.")
    YA = Y[0:N]
    YB = Y[N:2*N]
    V = float(np.var(np.r_[YA, YB], ddof=1))
    if not np.isfinite(V) or V <= 0:
        return {"S1": np.full(D, np.nan), "S1_conf": np.full(D, np.nan), "ST": np.full(D, np.nan), "ST_conf": np.full(D, np.nan)}
    S1 = np.zeros(D, dtype=float)
    ST = np.zeros(D, dtype=float)
    for j in range(D):
        YAB = Y[(2 + j) * N:(3 + j) * N]
        S1[j] = float(np.mean(YB * (YAB - YA)) / V)
        ST[j] = float(0.5 * np.mean((YA - YAB) ** 2) / V)
    return {"S1": S1, "S1_conf": np.full(D, np.nan), "ST": ST, "ST_conf": np.full(D, np.nan)}


def _sobol_analyze_one_vector(problem, Y):
    """Analyze one complete Sobol output vector using SALib or the internal fallback."""
    Y = np.asarray(Y, dtype=float)
    if not np.isfinite(Y).all():
        raise RuntimeError("Sobol output vector has NaN/inf values; inspect the saved outputs CSV first.")
    use_internal = globals().get("_SOBOL_SAMPLE_MODE") == "internal"
    if use_internal:
        return _internal_sobol_analyze(problem, Y, globals().get("_SOBOL_BASE_N_USED", SOBOL_BASE_N))
    _ensure_salib_for_sobol()
    return salib_sobol_analyze.analyze(problem, Y, calc_second_order=SOBOL_CALC_SECOND_ORDER, print_to_console=False)


def analyze_sobol_outputs(problem, outputs_df, label):
    output_cols = [
        "cumulative_exposure_attack_rate_pct",
        "peak_6hr_incidence_pct",
        "foi_peak",
        "foi_mean",
        "peak_active_infectious_pct",
    ]
    output_cols = [c for c in output_cols if c in outputs_df.columns]
    if not output_cols:
        return pd.DataFrame(columns=["sobol_label", "output", "parameter", "S1", "S1_sd", "S1_conf", "ST", "ST_sd", "ST_conf", "n_replicates"])

    # Analyze each stochastic replicate separately, then report mean ± SD across seeds.
    # This is more honest for a stochastic ABM than averaging the raw outputs first.
    by_seed_rows = []
    if "replicate_id" in outputs_df.columns:
        replicate_groups = list(outputs_df.groupby("replicate_id", sort=True))
    else:
        replicate_groups = [(0, outputs_df)]

    for rep_id, rep_df in replicate_groups:
        rep_df = rep_df.sort_values("sample_id").copy()
        seed_val = int(rep_df["seed"].iloc[0]) if "seed" in rep_df.columns and len(rep_df) else int(RUN_SEED)
        for out_col in output_cols:
            Y = pd.to_numeric(rep_df[out_col], errors="coerce").to_numpy(dtype=float)
            Si = _sobol_analyze_one_vector(problem, Y)
            for i, parameter in enumerate(problem["names"]):
                by_seed_rows.append({
                    "sobol_label": label,
                    "replicate_id": int(rep_id),
                    "seed": seed_val,
                    "output": out_col,
                    "parameter": parameter,
                    "S1": float(Si["S1"][i]) if np.isfinite(Si["S1"][i]) else np.nan,
                    "S1_conf": float(Si["S1_conf"][i]) if np.isfinite(Si["S1_conf"][i]) else np.nan,
                    "ST": float(Si["ST"][i]) if np.isfinite(Si["ST"][i]) else np.nan,
                    "ST_conf": float(Si["ST_conf"][i]) if np.isfinite(Si["ST_conf"][i]) else np.nan,
                })

    by_seed_df = pd.DataFrame(by_seed_rows)
    globals()[f"SOBOL_{str(label).upper()}_INDICES_BY_SEED"] = by_seed_df.copy()

    if by_seed_df.empty:
        return pd.DataFrame(columns=["sobol_label", "output", "parameter", "S1", "S1_sd", "S1_conf", "ST", "ST_sd", "ST_conf", "n_replicates"])

    grouped = by_seed_df.groupby(["sobol_label", "output", "parameter"], as_index=False)
    out = grouped.agg(
        S1=("S1", "mean"),
        S1_sd=("S1", "std"),
        S1_conf=("S1_conf", "mean"),
        ST=("ST", "mean"),
        ST_sd=("ST", "std"),
        ST_conf=("ST_conf", "mean"),
        n_replicates=("replicate_id", "nunique"),
    )
    out["S1_sd"] = out["S1_sd"].fillna(0.0)
    out["ST_sd"] = out["ST_sd"].fillna(0.0)
    return out

def _save_sobol_figures(indices_df, label, out_dir):
    if indices_df is None or indices_df.empty:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = list(dict.fromkeys(indices_df["parameter"].tolist()))
    outputs = list(dict.fromkeys(indices_df["output"].tolist()))

    def _nice_output_name(x):
        return {
            "cumulative_exposure_attack_rate_pct": "Cumulative exposure attack rate",
            "peak_6hr_incidence_pct": "Peak 6-hour incidence",
            "foi_peak": "FOI peak",
            "foi_mean": "FOI mean",
            "peak_active_infectious_pct": "Peak active infectious prevalence",
        }.get(x, str(x))

    for metric in ["S1", "ST"]:
        sd_col = f"{metric}_sd"
        mat = indices_df.pivot_table(index="parameter", columns="output", values=metric, aggfunc="mean").reindex(index=params, columns=outputs)
        sd_mat = (
            indices_df.pivot_table(index="parameter", columns="output", values=sd_col, aggfunc="mean")
            .reindex(index=params, columns=outputs)
            if sd_col in indices_df.columns else None
        )
        vals = mat.to_numpy(dtype=float)
        sds = sd_mat.to_numpy(dtype=float) if sd_mat is not None else np.zeros_like(vals)
        fig_w = max(6, 1.7 * len(outputs))
        fig_h = max(3.5, 0.55 * len(params) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        im = ax.imshow(vals, aspect="auto", vmin=0, vmax=max(1.0, np.nanmax(vals) if np.isfinite(vals).any() else 1.0))
        ax.set_xticks(np.arange(len(outputs)))
        ax.set_xticklabels([_nice_output_name(o) for o in outputs], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(params)))
        ax.set_yticklabels(params)
        ax.set_title(f"{label} Sobol {metric} heatmap")
        for r in range(vals.shape[0]):
            for c in range(vals.shape[1]):
                if np.isfinite(vals[r, c]):
                    sd_val = sds[r, c] if np.isfinite(sds[r, c]) else 0.0
                    ax.text(c, r, f"{vals[r, c]:.2f}\n±{sd_val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label=f"{metric} mean across seeds")
        fig.savefig(out_dir / f"fig_{CITY_SLUG}_sobol_{label}_{metric}_heatmap.png", dpi=220, bbox_inches="tight")
        fig.savefig(out_dir / f"fig_{CITY_SLUG}_sobol_{label}_{metric}_heatmap.pdf", bbox_inches="tight")
        plt.close(fig)

    n_outputs = len(outputs)
    fig, axs = plt.subplots(n_outputs, 1, figsize=(max(7, 1.0 * len(params)), max(3.0, 2.8 * n_outputs)), squeeze=False, constrained_layout=True)
    x = np.arange(len(params))
    width = 0.38
    for ax, output in zip(axs.ravel(), outputs):
        sub = indices_df[indices_df["output"] == output].set_index("parameter").reindex(params)
        s1 = pd.to_numeric(sub["S1"], errors="coerce").fillna(0).to_numpy(dtype=float)
        st = pd.to_numeric(sub["ST"], errors="coerce").fillna(0).to_numpy(dtype=float)
        s1_sd = pd.to_numeric(sub["S1_sd"], errors="coerce").fillna(0).to_numpy(dtype=float) if "S1_sd" in sub.columns else np.zeros_like(s1)
        st_sd = pd.to_numeric(sub["ST_sd"], errors="coerce").fillna(0).to_numpy(dtype=float) if "ST_sd" in sub.columns else np.zeros_like(st)
        ax.bar(x - width / 2, s1, width=width, label="S1", yerr=s1_sd, capsize=3)
        ax.bar(x + width / 2, st, width=width, label="ST", yerr=st_sd, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=25, ha="right")
        ymax = max(1.0, float(np.nanmax(np.r_[s1 + s1_sd, st + st_sd])) * 1.15 if len(params) else 1.0)
        ax.set_ylim(0, ymax)
        ax.set_title(_nice_output_name(output))
        ax.set_ylabel("Sobol index")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(f"{label} Sobol first-order and total indices (mean ± SD across seeds)", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / f"fig_{CITY_SLUG}_sobol_{label}_barplots.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"fig_{CITY_SLUG}_sobol_{label}_barplots.pdf", bbox_inches="tight")
    plt.close(fig)

def _run_and_save_sobol_block(label, problem, builder, sobol_out):
    samples, samples_df, outputs_df = run_sobol_experiment(
        label,
        problem,
        builder,
        base_n=SOBOL_BASE_N,
        replicate_seeds=SOBOL_REPLICATE_SEEDS,
        ticks=RUN_TICKS,
    )
    samples_df.to_csv(sobol_out / f"{CITY_SLUG}_sobol_{label}_samples.csv", index=False)
    outputs_df.to_csv(sobol_out / f"{CITY_SLUG}_sobol_{label}_outputs.csv", index=False)
    indices_df = analyze_sobol_outputs(problem, outputs_df, label)
    indices_df.to_csv(sobol_out / f"{CITY_SLUG}_sobol_{label}_indices.csv", index=False)
    by_seed_df = globals().get(f"SOBOL_{str(label).upper()}_INDICES_BY_SEED", None)
    if isinstance(by_seed_df, pd.DataFrame) and not by_seed_df.empty:
        by_seed_df.to_csv(sobol_out / f"{CITY_SLUG}_sobol_{label}_indices_by_seed.csv", index=False)
    _save_sobol_figures(indices_df, label, sobol_out)
    return {"samples": samples_df, "outputs": outputs_df, "indices": indices_df, "indices_by_seed": by_seed_df}

# ---- Trip diagnostics for baseline + selected mobility/D extremes ------------
RUN_TRIP_DIAGNOSTICS = True


def _build_zone_supply_payload(inner_src, generic_points, generic_weights, school_pts, market_pts, tap_pts, toilet_pts):
    """Diagnostics only: summarise destination supply by zone for gravity-response checks."""
    cols = ["poly_id", "generic_count", "generic_weight", "school_count", "market_count", "tap_count", "toilet_count"]
    if inner_src is None or (hasattr(inner_src, "empty") and inner_src.empty):
        return pd.DataFrame(columns=cols)
    df = inner_src[["poly_id", "geometry"]].copy() if "poly_id" in inner_src.columns else inner_src.reset_index(drop=True).assign(poly_id=lambda d: np.arange(len(d)))[["poly_id", "geometry"]]
    try:
        sidx = df.sindex
    except Exception:
        sidx = None
    poly_geoms = list(df.geometry)
    poly_ids = list(df["poly_id"].astype(int))
    def _pt_to_pid(pt):
        if pt is None:
            return None
        try:
            idxs = list(sidx.intersection(pt.bounds)) if sidx is not None else range(len(poly_geoms))
        except Exception:
            idxs = range(len(poly_geoms))
        for i in idxs:
            try:
                if poly_geoms[int(i)].contains(pt):
                    return poly_ids[int(i)]
            except Exception:
                continue
        return None
    rows = {int(pid): {"poly_id": int(pid), "generic_count": 0, "generic_weight": 0.0, "school_count": 0, "market_count": 0, "tap_count": 0, "toilet_count": 0} for pid in poly_ids}
    if generic_points:
        w = np.array(generic_weights, dtype=float) if generic_weights is not None and len(generic_weights)==len(generic_points) else np.ones(len(generic_points), dtype=float)
        for pt, ww in zip(generic_points, w):
            pid = _pt_to_pid(pt)
            if pid is not None:
                rows[int(pid)]["generic_count"] += 1
                rows[int(pid)]["generic_weight"] += float(ww)
    for pts, key in [(school_pts, "school_count"), (market_pts, "market_count"), (tap_pts, "tap_count"), (toilet_pts, "toilet_count")]:
        for pt in (pts or []):
            pid = _pt_to_pid(pt)
            if pid is not None:
                rows[int(pid)][key] += 1
    out = pd.DataFrame(list(rows.values())).sort_values("poly_id").reset_index(drop=True)
    return out

def _rebuild_full_trip_flows(result, fallback_inner_gdf=None, ticks_per_day=TICKS_PER_DAY):
    """
    Rebuild full OD flows (including within-zone stays) from snapshots.
    Uses result-specific inner polygons when available; otherwise falls back to
    the provided inner_gdf. This leaves simulation logic unchanged and is used
    only for diagnostics/figures.
    """
    inner_src = result.get("inner_polys", None)
    if inner_src is None or (hasattr(inner_src, "empty") and inner_src.empty):
        inner_src = fallback_inner_gdf
    tf_full, tfd_full, inner_used = _build_od_from_snapshots(result, inner_src, ticks_per_day=ticks_per_day)
    return tf_full, tfd_full, inner_used


def _trip_diag_metrics(tf_full, inner_used, n_agents, ticks, ticks_per_day=TICKS_PER_DAY):
    out = {
        "total_trips": 0.0,
        "trips_per_agent_per_day": np.nan,
        "average_trip_length_km": np.nan,
        "pct_trips_within_same_zone": np.nan,
        "n_zones": np.nan,
    }
    if tf_full is None or tf_full.empty:
        return out

    out["total_trips"] = float(tf_full["trips"].sum())
    days = max(1.0, float(ticks) / float(ticks_per_day))
    out["trips_per_agent_per_day"] = float(out["total_trips"] / max(int(n_agents), 1) / days)
    out["pct_trips_within_same_zone"] = float(100.0 * tf_full.loc[tf_full["origin_id"] == tf_full["dest_id"], "trips"].sum() / max(out["total_trips"], 1.0))

    if inner_used is not None and hasattr(inner_used, "empty") and not inner_used.empty:
        centroids = _ensure_inner_centroids(inner_used)
        dists = []
        wts = []
        for _, r in tf_full.iterrows():
            o = centroids.get(int(r["origin_id"]))
            d = centroids.get(int(r["dest_id"]))
            if o is None or d is None:
                continue
            dists.append(_dist_m(o, d) / 1000.0)
            wts.append(float(r["trips"]))
        if dists:
            out["average_trip_length_km"] = float(np.average(np.asarray(dists, dtype=float), weights=np.asarray(wts, dtype=float)))
        out["n_zones"] = int(len(inner_used))
    return out


def _pretty_tripdiag_label(scenario_tag):
    label_map = dict(SCENARIO_LABELS) if 'SCENARIO_LABELS' in globals() else {}
    # Final diagnostics use only the canonical core scenarios.
    return label_map.get(scenario_tag, str(scenario_tag))

def _save_trip_diagnostics(summary_df, od_matrices, heatmap_png, metrics_png):
    import math

    # metrics comparison figure (all diagnostic scenarios)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(summary_df))
    labels = [_pretty_tripdiag_label(s) for s in summary_df["scenario"].tolist()]

    axes[0].bar(x, summary_df["trips_per_agent_per_day"].to_numpy(dtype=float))
    axes[0].set_title("Trips / agent / day")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")

    axes[1].bar(x, summary_df["average_trip_length_km"].to_numpy(dtype=float))
    axes[1].set_title("Average trip length (km)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")

    axes[2].bar(x, summary_df["pct_trips_within_same_zone"].to_numpy(dtype=float))
    axes[2].set_title("% trips within same zone")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(metrics_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # O-D heatmaps figure (all diagnostic scenarios)
    n = len(od_matrices)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    share_mats = []
    for mat_df in od_matrices.values():
        mat = mat_df.to_numpy(dtype=float) if mat_df is not None and not mat_df.empty else np.zeros((1, 1), dtype=float)
        total = float(mat.sum())
        share = (mat / total) if total > 0 else mat
        if share.size:
            share_mats.append(share)
    vmax = max((float(np.nanmax(m)) for m in share_mats if m.size), default=1.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    for ax, (scenario_tag, mat_df) in zip(axes, od_matrices.items()):
        mat = mat_df.to_numpy(dtype=float) if mat_df is not None and not mat_df.empty else np.zeros((1, 1), dtype=float)
        total = float(mat.sum())
        share = (mat / total) if total > 0 else mat
        im = ax.imshow(share, aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_title(_pretty_tripdiag_label(scenario_tag))
        ax.set_xlabel("Destination zone")
        ax.set_ylabel("Origin zone")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(od_matrices):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(heatmap_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_tripdiagnostic_supervisor_plots(summary_df, od_matrices, out_dir):
    """Clean supervisor-facing plots without changing model logic."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _subset(order):
        df = summary_df.set_index("scenario", drop=False)
        rows = [df.loc[s] for s in order if s in df.index]
        return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=summary_df.columns)

    def _plot_metric_panel(df, out_png, title):
        if df.empty:
            return
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
        x = np.arange(len(df))
        labels = [_pretty_tripdiag_label(s) for s in df["scenario"].tolist()]
        specs = [
            ("trips_per_agent_per_day", "Trips per agent per day"),
            ("average_trip_length_km", "Average trip length (km)"),
            ("pct_trips_within_same_zone", "% trips within same zone"),
        ]
        for ax, (col, ttl) in zip(axes, specs):
            vals = df[col].to_numpy(dtype=float)
            ax.bar(x, vals)
            ax.set_title(ttl)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0, ha="center")
            ax.grid(axis="y", alpha=0.25)
            pad = max(0.02 * np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 0.0, 0.01)
            for xi, yi in zip(x, vals):
                if np.isfinite(yi):
                    ax.text(xi, yi + pad, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_od_triptych(order, out_png, title):
        mats = []
        for scenario in order:
            mat_df = od_matrices.get(scenario, pd.DataFrame())
            mat = mat_df.to_numpy(dtype=float) if mat_df is not None and not mat_df.empty else np.zeros((1, 1), dtype=float)
            total = float(mat.sum())
            share = (mat / total) if total > 0 else mat
            mats.append((scenario, share))
        vmax = max((float(np.nanmax(m)) for _, m in mats if m.size), default=1.0)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        fig, axes = plt.subplots(1, len(mats), figsize=(4.6 * max(1, len(mats)), 4.3))
        axes = np.atleast_1d(axes).ravel()
        last_im = None
        for ax, (scenario, share) in zip(axes, mats):
            last_im = ax.imshow(share, aspect="auto", vmin=0.0, vmax=vmax)
            ax.set_title(_pretty_tripdiag_label(scenario))
            ax.set_xlabel("Destination zone")
            ax.set_ylabel("Origin zone")
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.tolist(), fraction=0.025, pad=0.03, label="Trip share")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    core_order = SCENARIO_ORDER

    core_df = _subset(core_order)

    core_df.to_csv(out_dir / f"{CITY_SLUG}_trip_diagnostics_core_summary.csv", index=False)

    _plot_metric_panel(core_df, out_dir / f"fig_{CITY_SLUG}_trip_metrics_core_compare.png", "Core comparison: baseline vs current interventions")

    _plot_od_triptych(core_order, out_dir / f"fig_{CITY_SLUG}_od_heatmaps_core_compare.png", "Core O-D comparison: baseline vs current interventions")


def _build_inbound_by_zone(tf_full):
    if tf_full is None or tf_full.empty:
        return pd.DataFrame(columns=["poly_id", "inbound_trips"])
    inbound = tf_full.groupby("dest_id", as_index=False)["trips"].sum().rename(columns={"dest_id": "poly_id", "trips": "inbound_trips"})
    inbound["poly_id"] = inbound["poly_id"].astype(int)
    return inbound

def _merge_supply_inbound(result, tf_full):
    zone_supply = result.get("zone_supply", pd.DataFrame()).copy()
    if zone_supply is None or zone_supply.empty:
        return pd.DataFrame()
    inbound = _build_inbound_by_zone(tf_full)
    z = zone_supply.merge(inbound, how="left", on="poly_id").fillna({"inbound_trips": 0.0})
    z["supply_score"] = (z.get("generic_weight", 0.0).astype(float) + 0.5*z.get("school_count", 0).astype(float) + 0.5*z.get("market_count", 0).astype(float))
    return z

def _clean_exposure_source_label(source):
    s = str(source)
    # Plotting-only recode: household secondary exposure is grouped into
    # the existing Home water category so the stacked bar remains Tap water
    # versus Home water, where Home water includes household secondary spread.
    if "household" in s:
        return "Home water"
    if s == "home_water":
        return "Home water"
    if s == "tap_water":
        return "Tap water"
    if "contaminated_water" in s:
        return "Nearby contaminated water"
    return "Other / unknown"


def _save_gravity_response_plots(zone_results, out_prefix):
    core_order = [k for k in SCENARIO_ORDER if k in zone_results]
    if core_order:
        fig, axes = plt.subplots(1, len(core_order), figsize=(5.2 * len(core_order), 4.6))
        if len(core_order) == 1:
            axes = [axes]
        corr_rows = []
        dom_rows = []
        for ax, key in zip(axes, core_order):
            z = zone_results[key].copy()
            if z.empty:
                ax.set_visible(False)
                continue
            x = z["supply_score"].astype(float).to_numpy()
            y = z["inbound_trips"].astype(float).to_numpy()
            if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
                r = np.nan
            else:
                r = float(np.corrcoef(x, y)[0, 1])
            corr_rows.append({"scenario": key, "supply_inbound_corr": r})
            total_in = max(float(y.sum()), 1.0)
            dom_rows.append({"scenario": key, "dominant_zone_inbound_share_pct": float(100.0 * y.max() / total_in)})
            ax.scatter(x, y, alpha=0.7)
            if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all() and not np.allclose(np.std(x), 0):
                try:
                    coef = np.polyfit(x, y, 1)
                    xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                    # regression line intentionally omitted; correlations are diagnostic only
                    _ = (xx, coef)
                except Exception:
                    pass
            ax.set_title(f"{_pretty_tripdiag_label(key)}\\nr={r:.2f}" if np.isfinite(r) else _pretty_tripdiag_label(key))
            ax.set_xlabel("Zone supply score")
            ax.set_ylabel("Inbound trips")
        fig.suptitle("Gravity response: zone supply vs realised inbound trips")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_gravity_response_core.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(corr_rows).to_csv(f"{out_prefix}_gravity_response_core_corr.csv", index=False)
        pd.DataFrame(dom_rows).to_csv(f"{out_prefix}_dominant_zone_inbound_core.csv", index=False)

    core_bar_order = [k for k in SCENARIO_ORDER if k in zone_results]
    if core_bar_order:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        corr_vals = []
        dom_vals = []
        labels = []
        for key in core_bar_order:
            z = zone_results[key].copy()
            if z.empty:
                continue
            x = z["supply_score"].astype(float).to_numpy()
            y = z["inbound_trips"].astype(float).to_numpy()
            r = np.nan if (np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0)) else float(np.corrcoef(x, y)[0, 1])
            labels.append(_pretty_tripdiag_label(key).replace("\n", " "))
            corr_vals.append(r)
            dom_vals.append(float(100.0 * y.max() / max(float(y.sum()), 1.0)))
        axes[0].bar(np.arange(len(labels)), corr_vals)
        axes[0].set_xticks(np.arange(len(labels)))
        axes[0].set_xticklabels(labels, rotation=25, ha="right")
        axes[0].set_title("Supply→inbound correlation by scenario")
        axes[0].set_ylabel("Pearson r")
        axes[1].bar(np.arange(len(labels)), dom_vals)
        axes[1].set_xticks(np.arange(len(labels)))
        axes[1].set_xticklabels(labels, rotation=25, ha="right")
        axes[1].set_title("Dominant zone inbound share")
        axes[1].set_ylabel("% of all inbound trips")
        fig.suptitle("Gravity-model response diagnostics: baseline vs current interventions")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_gravity_response_core_bars.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, len(core_bar_order), figsize=(5.0 * len(core_bar_order), 4.8))
        if len(core_bar_order) == 1:
            axes = [axes]
        detail_rows = []
        for ax, key in zip(axes, core_bar_order):
            z = zone_results[key].copy()
            if z.empty:
                ax.set_visible(False)
                continue
            z = z[["poly_id", "supply_score", "inbound_trips"]].copy()
            z["supply_score"] = z["supply_score"].astype(float)
            z["inbound_trips"] = z["inbound_trips"].astype(float)
            x = z["supply_score"].to_numpy()
            y = z["inbound_trips"].to_numpy()
            pearson_r = np.nan if (np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0)) else float(np.corrcoef(x, y)[0, 1])
            sx = pd.Series(x).rank(method="average")
            sy = pd.Series(y).rank(method="average")
            spearman_r = np.nan if (np.allclose(np.std(sx), 0) or np.allclose(np.std(sy), 0)) else float(np.corrcoef(sx, sy)[0, 1])
            ax.scatter(x, y, alpha=0.7)
            if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all() and not np.allclose(np.std(x), 0):
                try:
                    coef = np.polyfit(x, y, 1)
                    xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                    # regression line intentionally omitted; correlations are diagnostic only
                    _ = (xx, coef)
                except Exception:
                    pass
            if len(z):
                dom = z.sort_values("inbound_trips", ascending=False).iloc[0]
                ax.annotate(f"zone {int(dom['poly_id'])}", (float(dom["supply_score"]), float(dom["inbound_trips"])), textcoords="offset points", xytext=(5, 5), fontsize=8)
            ax.set_title(_pretty_tripdiag_label(key))
            ax.set_xlabel("Zone supply score")
            ax.set_ylabel("Inbound trips")
            if np.isfinite(pearson_r) and np.isfinite(spearman_r):
                txt = f"Pearson r = {pearson_r:.2f}\nSpearman ρ = {spearman_r:.2f}"
            else:
                txt = "correlation undefined"
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
            detail_rows.append({"scenario": key, "pearson_r": pearson_r, "spearman_r": spearman_r})
        fig.suptitle("Zone supply vs realised inbound trips (baseline vs current interventions)")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_gravity_supply_inbound_scatter_core.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(detail_rows).to_csv(f"{out_prefix}_gravity_supply_inbound_scatter_core.csv", index=False)

        fig, axes = plt.subplots(len(core_bar_order), 1, figsize=(12, 3.0 * len(core_bar_order)), sharex=False)
        axes = np.atleast_1d(axes).ravel()
        bar_rows = []
        for ax, key in zip(axes, core_bar_order):
            z = zone_results[key].copy()
            if z.empty:
                ax.set_visible(False)
                continue
            z = z[["poly_id", "supply_score", "inbound_trips"]].copy()
            z["supply_score"] = z["supply_score"].astype(float)
            z["inbound_trips"] = z["inbound_trips"].astype(float)
            z["inbound_share_pct"] = 100.0 * z["inbound_trips"] / max(float(z["inbound_trips"].sum()), 1.0)
            z["supply_share_pct"] = 100.0 * z["supply_score"] / max(float(z["supply_score"].sum()), 1.0)
            z = z.sort_values("inbound_trips", ascending=False).reset_index(drop=True)
            topn = min(12, len(z))
            z = z.iloc[:topn].copy()
            xpos = np.arange(len(z))
            width = 0.42
            ax.bar(xpos - width / 2, z["inbound_share_pct"].to_numpy(dtype=float), width=width, label="Inbound trip share (%)")
            ax.bar(xpos + width / 2, z["supply_share_pct"].to_numpy(dtype=float), width=width, label="Supply share (%)")
            ax.set_xticks(xpos)
            ax.set_xticklabels([str(int(v)) for v in z["poly_id"]], rotation=0)
            ax.set_ylabel("% share")
            ax.set_title(_pretty_tripdiag_label(key))
            if ax is axes[0]:
                ax.legend(loc="upper right", fontsize=8)
            for _, rr in z.iterrows():
                bar_rows.append({"scenario": key, "poly_id": int(rr["poly_id"]), "inbound_share_pct": float(rr["inbound_share_pct"]), "supply_share_pct": float(rr["supply_share_pct"])})
        axes[-1].set_xlabel("Zone ID (sorted by inbound trips)")
        fig.suptitle("Sorted zone shares: realised inbound trips vs destination supply")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_gravity_sorted_zone_bars_core.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(bar_rows).to_csv(f"{out_prefix}_gravity_sorted_zone_bars_core.csv", index=False)


def _save_exposure_source_plots(exposure_results, out_prefix):
    raw_rows = []
    grouped_rows = []
    for key, result in exposure_results.items():
        ex = result.get("exposures", pd.DataFrame())
        if ex is None or ex.empty or "source" not in ex.columns:
            continue
        n_agents = max(int(result.get("N", len(agent_list))), 1)
        tmp = ex.copy()

        raw = tmp["source"].fillna("unknown").astype(str).value_counts().rename_axis("source").reset_index(name="count")
        raw["scenario"] = key
        raw["N"] = n_agents
        raw["attack_rate_pct"] = 100.0 * raw["count"].astype(float) / float(n_agents)
        raw_rows.append(raw)

        tmp["source_clean"] = tmp["source"].fillna("unknown").apply(_clean_exposure_source_label)
        grouped = tmp["source_clean"].value_counts().rename_axis("source").reset_index(name="count")
        grouped["scenario"] = key
        grouped["N"] = n_agents
        grouped["attack_rate_pct"] = 100.0 * grouped["count"].astype(float) / float(n_agents)
        grouped_rows.append(grouped)
    if not grouped_rows:
        return

    raw_df = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame(columns=["source", "count", "scenario", "N", "attack_rate_pct"])
    ex_df = pd.concat(grouped_rows, ignore_index=True)
    raw_df.to_csv(f"{out_prefix}_exposure_source_raw_counts.csv", index=False)
    ex_df.to_csv(f"{out_prefix}_exposure_source_counts.csv", index=False)

    def _ordered_pivot(df, order, value_col):
        pivot = df.pivot_table(index="scenario", columns="source", values=value_col, aggfunc="sum", fill_value=0)
        pivot = pivot.reindex([k for k in order if k in pivot.index]).fillna(0)
        return pivot

    def _plot_grouped(df, order, fname, title, value_col, ylabel):
        pivot = _ordered_pivot(df, order, value_col)
        if pivot.empty:
            return
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_xticklabels([_pretty_tripdiag_label(s).replace("\n", " ") for s in pivot.index], rotation=15, ha="right")
        ax.legend(title="source", fontsize=8)
        plt.tight_layout()
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()

    def _plot_stacked_pct(df, order, fname, title):
        pivot = _ordered_pivot(df, order, "count")
        if pivot.empty:
            return
        pct = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
        ax = pct.plot(kind="bar", stacked=True, figsize=(10, 5))
        ax.set_title(title)
        ax.set_ylabel("% of logged S→E exposures")
        ax.set_xlabel("")
        ax.set_xticklabels([_pretty_tripdiag_label(s).replace("\n", " ") for s in pct.index], rotation=15, ha="right")
        ax.legend(title="source", fontsize=8)
        plt.tight_layout()
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()

    def _plot_counts_and_pct_side_by_side(df, order, fname_png, fname_pdf, title):
        pivot = _ordered_pivot(df, order, "count")
        if pivot.empty:
            return
        pct = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
        labels = [_pretty_tripdiag_label(s).replace("\n", " ") for s in pivot.index]
        fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
        pivot.plot(kind="bar", ax=axs[0])
        axs[0].set_title("Absolute S→E counts by source")
        axs[0].set_ylabel("Logged S→E exposures")
        axs[0].set_xlabel("")
        axs[0].set_xticklabels(labels, rotation=15, ha="right")
        axs[0].grid(axis="y", alpha=0.25)
        axs[0].legend(title="source", fontsize=8)

        pct.plot(kind="bar", stacked=True, ax=axs[1])
        axs[1].set_title("Percentage source composition")
        axs[1].set_ylabel("% of logged S→E exposures")
        axs[1].set_xlabel("")
        axs[1].set_xticklabels(labels, rotation=15, ha="right")
        axs[1].set_ylim(0, 100)
        axs[1].grid(axis="y", alpha=0.25)
        axs[1].legend(title="source", fontsize=8)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(fname_png, dpi=220, bbox_inches="tight")
        fig.savefig(fname_pdf, bbox_inches="tight")
        plt.close(fig)

    core_order = SCENARIO_ORDER
    # Main plots: absolute and agent-normalised source burden. These show whether
    # interventions actually reduce exposure, instead of forcing each scenario to 100%.
    _plot_grouped(ex_df, core_order, f"{out_prefix}_exposure_source_absolute_core.png", "Exposure sources: absolute S→E counts", "count", "Logged S→E exposures")
    _plot_grouped(ex_df, core_order, f"{out_prefix}_exposure_source_attack_rate_core.png", "Exposure sources: S→E attack rate by source", "attack_rate_pct", "S→E exposures (% of agents)")
    _plot_grouped(raw_df, core_order, f"{out_prefix}_exposure_source_raw_absolute_core.png", "Raw exposure sources: absolute S→E counts", "count", "Logged S→E exposures")
    _plot_grouped(raw_df, core_order, f"{out_prefix}_exposure_source_raw_attack_rate_core.png", "Raw exposure sources: S→E attack rate by source", "attack_rate_pct", "S→E exposures (% of agents)")

    # Combined figure: exposure counts and source composition.
    # Existing absolute/attack-rate/composition filenames above are retained.
    _plot_counts_and_pct_side_by_side(
        ex_df,
        core_order,
        f"{out_prefix}_exposure_source_counts_and_pct_core.png",
        f"{out_prefix}_exposure_source_counts_and_pct_core.pdf",
        "Exposure sources: absolute burden and source composition",
    )
    _plot_counts_and_pct_side_by_side(
        raw_df,
        core_order,
        f"{out_prefix}_exposure_source_raw_counts_and_pct_core.png",
        f"{out_prefix}_exposure_source_raw_counts_and_pct_core.pdf",
        "Raw exposure sources: absolute burden and source composition",
    )

    # Secondary composition-only diagnostics retained for comparison with earlier runs.
    _plot_stacked_pct(ex_df, core_order, f"{out_prefix}_exposure_source_composition_core.png", "Exposure source composition: baseline vs current interventions")
    _plot_stacked_pct(raw_df, core_order, f"{out_prefix}_exposure_source_raw_composition_core.png", "Raw exposure source composition diagnostics: baseline vs current interventions")


def _facility_visit_burden_from_result(result, ticks, ticks_per_day=TICKS_PER_DAY):
    fv = result.get("facility_visits", pd.DataFrame())
    if fv is not None and not getattr(fv, "empty", True) and "facility_type" in fv.columns:
        days = max(1.0, float(ticks) / float(ticks_per_day))
        n_agents = max(int(result.get("N", len(agent_list))), 1)
        tap_fv = fv[fv["facility_type"] == "tap"].copy()
        toilet_fv = fv[fv["facility_type"] == "toilet"].copy()
        contaminated_taps = set(result.get("contaminated_tap_wkbs", []))
        def _counts(frame):
            if frame.empty or "facility_wkb" not in frame.columns:
                return Counter()
            return Counter(frame["facility_wkb"].dropna().tolist())
        tap_counts = _counts(tap_fv)
        toilet_counts = _counts(toilet_fv)
        def _top_share(counter_obj, k):
            total = float(sum(counter_obj.values()))
            if total <= 0:
                return np.nan
            top = sum(v for _, v in counter_obj.most_common(k))
            return float(100.0 * top / total)
        def _gini_from_counter(counter_obj):
            vals = np.asarray(list(counter_obj.values()), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0 or float(vals.sum()) <= 0:
                return np.nan
            vals = np.sort(vals)
            n = vals.size
            return float((2.0 * np.sum((np.arange(1, n + 1)) * vals) / (n * vals.sum())) - ((n + 1.0) / n))
        tap_total = float(len(tap_fv))
        toilet_total = float(len(toilet_fv))
        contaminated_visits = float(sum(v for k, v in tap_counts.items() if k in contaminated_taps))
        return {
            "tap_total_visits": int(tap_total),
            "toilet_total_visits": int(toilet_total),
            "contaminated_tap_count": int(len(contaminated_taps)),
            "contaminated_tap_visit_count": int(contaminated_visits),
            "contaminated_tap_visit_share_pct": float(100.0 * contaminated_visits / tap_total) if tap_total > 0 else np.nan,
            "tap_visits_per_agent_per_day": float(tap_total / n_agents / days),
            "tap_share_top1_pct": _top_share(tap_counts, 1),
            "tap_share_top3_pct": _top_share(tap_counts, 3),
            "tap_visit_gini": _gini_from_counter(tap_counts),
            "tap_unique_visited": int(len(tap_counts)),
            "tap_avg_agents_per_tap": float(tap_fv.groupby("facility_wkb")["agent"].nunique().mean()) if (not tap_fv.empty and "facility_wkb" in tap_fv.columns) else np.nan,
            "tap_avg_visits_per_contaminated_tap": float(contaminated_visits / max(len(contaminated_taps), 1)) if contaminated_taps else np.nan,
            "toilet_visits_per_agent_per_day": float(toilet_total / n_agents / days),
            "toilet_share_top1_pct": _top_share(toilet_counts, 1),
            "toilet_share_top3_pct": _top_share(toilet_counts, 3),
            "toilet_unique_used": int(len(toilet_counts)),
            "toilet_avg_agents_per_toilet": float(toilet_fv.groupby("facility_wkb")["agent"].nunique().mean()) if (not toilet_fv.empty and "facility_wkb" in toilet_fv.columns) else np.nan,
        }
    snaps = result.get("snapshots", [])
    tap_points_local = list(result.get("tap_points_active", []))
    toilet_points_local = list(result.get("toilet_geoms_active", []))
    contaminated_taps = set(result.get("contaminated_tap_wkbs", []))
    tap_wkbs = {getattr(pt, "wkb", None) for pt in tap_points_local if pt is not None}
    toilet_wkbs = {getattr(pt, "wkb", None) for pt in toilet_points_local if pt is not None}
    tap_counts = Counter()
    toilet_counts = Counter()
    tap_agents = defaultdict(set)
    toilet_agents = defaultdict(set)
    for snap in snaps:
        if snap is None or len(snap) == 0:
            continue
        cols_needed = [c for c in ["unique_id", "current_location"] if c in snap.columns]
        if len(cols_needed) < 2:
            continue
        for _, row in snap[cols_needed].iterrows():
            loc = row["current_location"]
            uid = int(row["unique_id"])
            try:
                wkb = loc.wkb
            except Exception:
                continue
            if wkb in tap_wkbs:
                tap_counts[wkb] += 1
                tap_agents[wkb].add(uid)
            if wkb in toilet_wkbs:
                toilet_counts[wkb] += 1
                toilet_agents[wkb].add(uid)
    days = max(1.0, float(ticks) / float(ticks_per_day))
    n_agents = max(int(result.get("N", len(agent_list))), 1)

    def _top_share(counter_obj, k):
        total = float(sum(counter_obj.values()))
        if total <= 0:
            return np.nan
        top = sum(v for _, v in counter_obj.most_common(k))
        return float(100.0 * top / total)

    def _gini_from_counter(counter_obj):
        vals = np.asarray(list(counter_obj.values()), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0 or float(vals.sum()) <= 0:
            return np.nan
        vals = np.sort(vals)
        n = vals.size
        return float((2.0 * np.sum((np.arange(1, n + 1)) * vals) / (n * vals.sum())) - ((n + 1.0) / n))

    tap_total = float(sum(tap_counts.values()))
    toilet_total = float(sum(toilet_counts.values()))
    contaminated_visits = float(sum(v for k, v in tap_counts.items() if k in contaminated_taps))
    return {
        "tap_total_visits": int(tap_total),
        "toilet_total_visits": int(toilet_total),
        "contaminated_tap_count": int(len(contaminated_taps)),
        "contaminated_tap_visit_count": int(contaminated_visits),
        "contaminated_tap_visit_share_pct": float(100.0 * contaminated_visits / tap_total) if tap_total > 0 else np.nan,
        "tap_visits_per_agent_per_day": float(tap_total / n_agents / days),
        "tap_share_top1_pct": _top_share(tap_counts, 1),
        "tap_share_top3_pct": _top_share(tap_counts, 3),
        "tap_visit_gini": _gini_from_counter(tap_counts),
        "tap_unique_visited": int(sum(1 for v in tap_counts.values() if v > 0)),
        "tap_avg_agents_per_tap": float(np.mean([len(v) for v in tap_agents.values()])) if tap_agents else np.nan,
        "tap_avg_visits_per_contaminated_tap": float(contaminated_visits / max(len(contaminated_taps), 1)) if contaminated_taps else np.nan,
        "toilet_visits_per_agent_per_day": float(toilet_total / n_agents / days),
        "toilet_share_top1_pct": _top_share(toilet_counts, 1),
        "toilet_share_top3_pct": _top_share(toilet_counts, 3),
        "toilet_unique_used": int(sum(1 for v in toilet_counts.values() if v > 0)),
        "toilet_avg_agents_per_toilet": float(np.mean([len(v) for v in toilet_agents.values()])) if toilet_agents else np.nan,
    }


def _save_facility_burden_plots(burden_df, out_prefix):
    if burden_df is None or burden_df.empty:
        return
    burden_df.to_csv(f"{out_prefix}_facility_burden_summary.csv", index=False)

    def _plot(order, fname, title):
        sub = burden_df[burden_df["scenario"].isin(order)].copy()
        cats = [k for k in order if k in sub["scenario"].unique()]
        sub["scenario"] = pd.Categorical(sub["scenario"], categories=cats, ordered=True)
        sub = sub.sort_values("scenario")
        if sub.empty:
            return
        metrics = [
            ("tap_visits_per_agent_per_day", "Tap visits / agent / day"),
            ("tap_share_top3_pct", "Tap visits to top 3 taps (%)"),
            ("tap_visit_gini", "Tap visit Gini"),
            ("tap_unique_visited", "Unique taps visited"),
            ("tap_avg_agents_per_tap", "Average agents per tap"),
            ("tap_avg_visits_per_contaminated_tap", "Mean burden per contaminated tap"),
            ("toilet_visits_per_agent_per_day", "Toilet visits / agent / day"),
        ]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for ax, (col, ttl) in zip(axes.flat, metrics):
            vals = sub[col].to_numpy(dtype=float) if col in sub.columns else np.full(len(sub), np.nan)
            x = np.arange(len(sub))
            ax.bar(x, vals)
            ax.set_xticks(x)
            ax.set_xticklabels([_pretty_tripdiag_label(s).replace("\n", " ") for s in sub["scenario"]], rotation=15, ha="right")
            ax.set_title(ttl)
        for ax in axes.flat[len(metrics):]:
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    _core_order = SCENARIO_ORDER
    _plot(_core_order, f"{out_prefix}_facility_burden_core.png", "Local facility burden: baseline vs interventions")

    # Absolute counts companion plot: keeps the rate/percentage panels interpretable.
    _sub = burden_df[burden_df["scenario"].isin(_core_order)].copy()
    _cats = [k for k in _core_order if k in _sub["scenario"].unique()]
    if _cats:
        _sub["scenario"] = pd.Categorical(_sub["scenario"], categories=_cats, ordered=True)
        _sub = _sub.sort_values("scenario")
        _abs_metrics = [
            ("tap_total_visits", "Total tap visits"),
            ("toilet_total_visits", "Total toilet visits"),
            ("contaminated_tap_visit_count", "Visits to contaminated taps"),
            ("contaminated_tap_visit_share_pct", "% tap visits to contaminated taps"),
            ("contaminated_tap_count", "Contaminated taps"),
        ]
        fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
        for ax, (col, ttl) in zip(axes.flat, _abs_metrics):
            vals = _sub[col].to_numpy(dtype=float) if col in _sub.columns else np.full(len(_sub), np.nan)
            x = np.arange(len(_sub))
            ax.bar(x, vals)
            ax.set_xticks(x)
            ax.set_xticklabels([_pretty_tripdiag_label(s).replace("\n", " ") for s in _sub["scenario"]], rotation=20, ha="right")
            ax.set_title(ttl)
        fig.suptitle("Local facility burden: absolute counts")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_facility_burden_absolute_counts_core.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

def _run_seed_variability(seed_values, ticks, scenarios):
    global SEED_DAILY_INCIDENCE_ROWS, SEED_ACTIVE_INFECTION_ROWS, SEED_6HR_INCIDENCE_ROWS, SEED_6HR_ACTIVE_INFECTION_ROWS, SEED_PEAK_6HR_INCIDENCE_ROWS
    SEED_DAILY_INCIDENCE_ROWS = []
    SEED_ACTIVE_INFECTION_ROWS = []
    SEED_6HR_INCIDENCE_ROWS = []
    SEED_6HR_ACTIVE_INFECTION_ROWS = []
    SEED_PEAK_6HR_INCIDENCE_ROWS = []
    rows=[]
    for seed in seed_values:
        for tag, intervention in scenarios.items():
            iv = None if intervention is None else dict(intervention, seed=int(seed))
            res = run_scenario(ticks=ticks, seed=int(seed), intervention=iv)
            tf_full, _, inner_used = _rebuild_full_trip_flows(res, fallback_inner_gdf=res.get("inner_gdf_plot", inner_gdf), ticks_per_day=TICKS_PER_DAY)
            m = _trip_diag_metrics(tf_full, inner_used, n_agents=res.get("N", len(agent_list)), ticks=ticks, ticks_per_day=TICKS_PER_DAY)
            hist = res.get("state_history", pd.DataFrame())
            if hist is not None and not hist.empty:
                # Keep the old daily summaries as secondary CSV diagnostics.
                daily_inc = daily_incidence_percent_from_transitions(res.get("transitions", pd.DataFrame()), ticks_per_day=TICKS_PER_DAY)
                for _day, _inc in daily_inc.items():
                    SEED_DAILY_INCIDENCE_ROWS.append({"seed": int(seed), "scenario": tag, "day": int(_day), "daily_incidence_pct": float(_inc)})
                active_prev = active_infectious_percent(hist, ticks_per_day=TICKS_PER_DAY)
                for _day, _prev in active_prev.items():
                    SEED_ACTIVE_INFECTION_ROWS.append({"seed": int(seed), "scenario": tag, "day": int(_day), "active_infectious_pct": float(_prev)})

                # Main publication-style time series: 6-hour tick bins.
                inc_6h = six_hour_incidence_percent_from_transitions(res.get("transitions", pd.DataFrame()), bin_ticks=6)
                for _, _r in inc_6h.iterrows():
                    SEED_6HR_INCIDENCE_ROWS.append({
                        "seed": int(seed), "scenario": tag,
                        "tick_start": int(_r["tick_start"]), "tick_end": int(_r["tick_end"]), "tick_mid": float(_r["tick_mid"]),
                        "incidence_pct": float(_r["incidence_pct"]),
                    })
                active_6h = six_hour_active_infectious_percent(hist, bin_ticks=6)
                for _, _r in active_6h.iterrows():
                    SEED_6HR_ACTIVE_INFECTION_ROWS.append({
                        "seed": int(seed), "scenario": tag,
                        "tick_start": int(_r["tick_start"]), "tick_end": int(_r["tick_end"]), "tick_mid": float(_r["tick_mid"]),
                        "active_infectious_pct": float(_r["active_infectious_pct"]),
                    })
                if inc_6h is not None and not inc_6h.empty:
                    _peak_row = inc_6h.loc[inc_6h["incidence_pct"].astype(float).idxmax()]
                    peak_6hr_inc = float(_peak_row["incidence_pct"])
                    peak_tick_start = int(_peak_row["tick_start"])
                    peak_tick_end = int(_peak_row["tick_end"])
                else:
                    peak_6hr_inc = np.nan
                    peak_tick_start = -1
                    peak_tick_end = -1
                SEED_PEAK_6HR_INCIDENCE_ROWS.append({
                    "seed": int(seed), "scenario": tag,
                    "peak_6hr_incidence_pct": peak_6hr_inc,
                    "peak_tick_start": peak_tick_start,
                    "peak_tick_end": peak_tick_end,
                })
                peak_active_i = float(active_6h["active_infectious_pct"].max()) if active_6h is not None and not active_6h.empty else np.nan
            else:
                peak_6hr_inc = np.nan
                peak_active_i = np.nan
            rows.append({"seed": int(seed), "scenario": tag, "peak_6hr_incidence_pct": peak_6hr_inc, "peak_active_infectious_pct": peak_active_i, **m})
    return pd.DataFrame(rows)

def _save_seed_variability_plots(seed_df, out_prefix):
    if seed_df is None or seed_df.empty:
        return
    seed_df.to_csv(f"{out_prefix}_seed_variability.csv", index=False)
    order = [k for k in SCENARIO_ORDER if k in seed_df["scenario"].unique()]
    agg = seed_df.groupby("scenario", as_index=False).agg({
        "trips_per_agent_per_day": ["mean", "std"],
        "average_trip_length_km": ["mean", "std"],
        "pct_trips_within_same_zone": ["mean", "std"],
        "peak_6hr_incidence_pct": ["mean", "std"],
        "peak_active_infectious_pct": ["mean", "std"],
    })
    agg.columns = ["scenario"] + [f"{a}_{b}" for a,b in agg.columns.tolist()[1:]]
    agg = agg.set_index("scenario").reindex(order).reset_index()
    metrics = [
        ("trips_per_agent_per_day", "Trips / agent / day"),
        ("average_trip_length_km", "Average trip length (km)"),
        ("pct_trips_within_same_zone", "% trips within same zone"),
        ("peak_6hr_incidence_pct", "Peak 6-hour incidence (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.flat, metrics):
        x = np.arange(len(agg))
        _means = agg[f"{col}_mean"].to_numpy(dtype=float)
        _errs = agg[f"{col}_std"].fillna(0).to_numpy(dtype=float)
        ax.bar(x, _means, yerr=_errs, capsize=4)
        if col == "peak_6hr_incidence_pct" and np.isfinite(_means).any():
            _lo = float(np.nanmin(_means - _errs))
            _hi = float(np.nanmax(_means + _errs))
            _span = max(_hi - _lo, 1e-6)
            _pad = max(0.15 * _span, 0.05)
            ax.set_ylim(max(0.0, _lo - _pad), _hi + _pad)
            ax.text(0.02, 0.96, "zoomed y-axis", transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_tripdiag_label(s).replace("\n", " ") for s in agg["scenario"]], rotation=15, ha='right')
        ax.set_title(title)
    fig.suptitle("Across-seed variability diagnostics")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_seed_variability_core.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Secondary old daily incidence CSV only, kept for traceability.
    daily_rows = globals().get("SEED_DAILY_INCIDENCE_ROWS", [])
    if daily_rows:
        pd.DataFrame(daily_rows).to_csv(f"{out_prefix}_seed_daily_incidence.csv", index=False)

    # Main 6-hour incidence mean ± SD across seed runs. X-axis is tick number.
    inc6_rows = globals().get("SEED_6HR_INCIDENCE_ROWS", [])
    if inc6_rows:
        inc6_df = pd.DataFrame(inc6_rows)
        inc6_df.to_csv(f"{out_prefix}_seed_6hr_incidence.csv", index=False)
        agg_inc6 = inc6_df.groupby(["scenario", "tick_mid"], as_index=False)["incidence_pct"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for scenario in order:
            sub = agg_inc6[agg_inc6["scenario"] == scenario].sort_values("tick_mid")
            if sub.empty:
                continue
            x = sub["tick_mid"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            sd = sub["std"].fillna(0).to_numpy(dtype=float)
            ax.plot(x, y, linewidth=2, label=_pretty_tripdiag_label(scenario).replace("\n", " "))
            ax.fill_between(x, y - sd, y + sd, alpha=0.15)
        ax.set_xlabel("Tick")
        ax.set_ylabel("6-hour incidence (%)")
        ax.set_title("6-hour incidence across seeds: mean ± SD")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_seed_6hr_incidence_core.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Peak 6-hour incidence per seed traceability table.
    peak6_rows = globals().get("SEED_PEAK_6HR_INCIDENCE_ROWS", [])
    if peak6_rows:
        pd.DataFrame(peak6_rows).to_csv(f"{out_prefix}_seed_peak_6hr_incidence.csv", index=False)

    # Secondary old daily active prevalence CSV only, kept for traceability.
    active_rows = globals().get("SEED_ACTIVE_INFECTION_ROWS", [])
    if active_rows:
        pd.DataFrame(active_rows).to_csv(f"{out_prefix}_seed_active_infectious_prevalence.csv", index=False)

    # Main 6-hour active infectious prevalence mean ± SD. X-axis is tick number.
    active6_rows = globals().get("SEED_6HR_ACTIVE_INFECTION_ROWS", [])
    if active6_rows:
        active6_df = pd.DataFrame(active6_rows)
        active6_df.to_csv(f"{out_prefix}_seed_6hr_active_infectious_prevalence.csv", index=False)
        agg_active6 = active6_df.groupby(["scenario", "tick_mid"], as_index=False)["active_infectious_pct"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for scenario in order:
            sub = agg_active6[agg_active6["scenario"] == scenario].sort_values("tick_mid")
            if sub.empty:
                continue
            x = sub["tick_mid"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            sd = sub["std"].fillna(0).to_numpy(dtype=float)
            ax.plot(x, y, linewidth=2, label=_pretty_tripdiag_label(scenario).replace("\n", " "))
            ax.fill_between(x, y - sd, y + sd, alpha=0.15)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Active infectious prevalence (% I)")
        ax.set_title("6-hour active infectious prevalence across seeds: mean ± SD")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_seed_6hr_active_infectious_prevalence_core.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Monotone infection-related outcome: cumulative S→E attack rate. This is
    # the clearest "goes up" plot and separates interventions because it uses
    # the transition ledger rather than active I, which is delayed by E→I.
    if inc6_rows:
        inc6_df = pd.DataFrame(inc6_rows).sort_values(["scenario", "seed", "tick_mid"])
        inc6_df["cumulative_exposure_pct"] = inc6_df.groupby(["scenario", "seed"])["incidence_pct"].cumsum()
        inc6_df.to_csv(f"{out_prefix}_seed_6hr_cumulative_exposure_attack_rate.csv", index=False)
        agg_attack = inc6_df.groupby(["scenario", "tick_mid"], as_index=False)["cumulative_exposure_pct"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for scenario in order:
            sub = agg_attack[agg_attack["scenario"] == scenario].sort_values("tick_mid")
            if sub.empty:
                continue
            x = sub["tick_mid"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            sd = sub["std"].fillna(0).to_numpy(dtype=float)
            ax.plot(x, y, linewidth=2.5, label=_pretty_tripdiag_label(scenario).replace("\n", " "))
            ax.fill_between(x, y - sd, y + sd, alpha=0.12)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Cumulative S→E exposures (% of agents)")
        ax.set_title("6-hour cumulative exposure attack rate across seeds: mean ± SD")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_seed_6hr_cumulative_exposure_attack_rate_core.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

if RUN_TRIP_DIAGNOSTICS:
    diagnostic_results = {
        "baseline": base,
        "intervB_mobilityreduce": res_B,
        "intervD_destinations_only": globals().get("res_D_dest", None),
        "intervD_facilities_only": globals().get("res_D_fac", None),
        "intervD_area_redistribute": res_D,
    }
    diagnostic_results = {k: v for k, v in diagnostic_results.items() if v is not None}

    def _actual_trip_metrics_from_result(result, base_metrics):
        out = dict(base_metrics)
        at = result.get("actual_trips", pd.DataFrame())
        if at is not None and not getattr(at, "empty", True) and "actual_trip_distance_m" in at.columns:
            days = max(1.0, float(RUN_TICKS) / float(TICKS_PER_DAY))
            n_agents = max(int(result.get("N", len(agent_list))), 1)
            moved = at[at["actual_trip_distance_m"].astype(float) > 0].copy()
            out["actual_logged_steps"] = int(len(at))
            out["actual_moved_steps"] = int(len(moved))
            out["actual_average_step_length_km"] = float(moved["actual_trip_distance_m"].astype(float).mean() / 1000.0) if len(moved) else 0.0
            out["actual_steps_per_agent_per_day"] = float(len(moved) / n_agents / days)
            if {"origin_zone", "destination_zone"}.issubset(at.columns):
                denom = max(len(at), 1)
                out["actual_pct_steps_within_same_zone"] = float(100.0 * (at["origin_zone"] == at["destination_zone"]).sum() / denom)
            # Use actual movement for the headline average trip length, while retaining OD-derived columns above.
            out["average_trip_length_km"] = out["actual_average_step_length_km"]
        return out

    diag_rows = []
    od_matrices = {}
    for scenario_tag, result in diagnostic_results.items():
        tf_full, tfd_full, inner_used = _rebuild_full_trip_flows(result, fallback_inner_gdf=result.get("inner_gdf_plot", inner_gdf), ticks_per_day=TICKS_PER_DAY)
        metrics = _trip_diag_metrics(tf_full, inner_used, n_agents=result.get("N", len(agent_list)), ticks=RUN_TICKS, ticks_per_day=TICKS_PER_DAY)
        metrics = _actual_trip_metrics_from_result(result, metrics)
        diag_rows.append({"scenario": scenario_tag, **metrics})

        tf_out = os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_tripflows_full.csv")
        tfd_out = os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_tripflows_full_daily.csv")
        if tf_full is not None:
            tf_full.to_csv(tf_out, index=False)
        if tfd_full is not None:
            tfd_full.to_csv(tfd_out, index=False)
        _at_full = result.get("actual_trips", pd.DataFrame())
        if _at_full is not None and hasattr(_at_full, "to_csv"):
            _at_full.to_csv(os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_actual_trips_full.csv"), index=False)
        _fv_full = result.get("facility_visits", pd.DataFrame())
        if _fv_full is not None and hasattr(_fv_full, "to_csv"):
            _fv_full.to_csv(os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_facility_visits_full.csv"), index=False)

        if tf_full is not None and not tf_full.empty:
            od_mat = (
                tf_full.pivot_table(index="origin_id", columns="dest_id", values="trips", aggfunc="sum", fill_value=0)
                .sort_index(axis=0)
                .sort_index(axis=1)
            )
        else:
            od_mat = pd.DataFrame()
        od_matrices[scenario_tag] = od_mat
        od_out = os.path.join(SHARED_OUT, f"{CITY_SLUG}_{scenario_tag}_od_matrix.csv")
        od_mat.to_csv(od_out, index=True)

    trip_diag_summary_df = pd.DataFrame(diag_rows)
    trip_diag_summary_path = os.path.join(SHARED_OUT, f"{CITY_SLUG}_trip_diagnostics_summary.csv")
    trip_diag_summary_df.to_csv(trip_diag_summary_path, index=False)

    metrics_png = os.path.join(CITY_OUT, f"fig_{CITY_SLUG}_trip_metrics_compare.png")
    heatmap_png = os.path.join(CITY_OUT, f"fig_{CITY_SLUG}_od_heatmaps.png")
    _save_trip_diagnostics(trip_diag_summary_df, od_matrices, heatmap_png=heatmap_png, metrics_png=metrics_png)
    _save_tripdiagnostic_supervisor_plots(trip_diag_summary_df, od_matrices, CITY_OUT)

    # Actual within-zone distance distribution: separates true no-movement from same-zone movement.
    try:
        _dist_rows = []
        for _tag, _res in diagnostic_results.items():
            _at = _res.get("actual_trips", pd.DataFrame())
            if _at is None or getattr(_at, "empty", True):
                continue
            _tmp = _at.copy()
            _tmp = _tmp[_tmp["origin_zone"] == _tmp["destination_zone"]]
            _tmp = _tmp[_tmp["actual_trip_distance_m"].astype(float) > 0]
            if _tmp.empty:
                continue
            _tmp["scenario"] = _tag
            _dist_rows.append(_tmp[["scenario", "actual_trip_distance_m", "actual_moved_distance_bin"]])
        if _dist_rows:
            _dist_df = pd.concat(_dist_rows, ignore_index=True)
            _dist_df.to_csv(os.path.join(SHARED_OUT, f"{CITY_SLUG}_within_zone_actual_distance_distribution.csv"), index=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            _plot_df = _dist_df.copy()
            _plot_df["actual_trip_distance_km"] = _plot_df["actual_trip_distance_m"].astype(float) / 1000.0
            _labels = _canonical_scenario_order(_plot_df["scenario"].unique())
            data = [_plot_df.loc[_plot_df["scenario"] == k, "actual_trip_distance_km"].to_numpy(dtype=float) for k in _labels]
            ax.boxplot(data, labels=[_pretty_tripdiag_label(k).replace("\n", " ") for k in _labels], showfliers=False)
            ax.set_ylabel("Within-zone actual movement distance (km)")
            ax.set_title("Within-zone movement distances: diagonal OD does not mean no movement")
            ax.tick_params(axis='x', rotation=15)
            fig.tight_layout()
            fig.savefig(os.path.join(CITY_OUT, f"fig_{CITY_SLUG}_within_zone_actual_distance_distribution.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
    except Exception as _e_distplot:
        print("[trip diagnostics] within-zone distance plot skipped:", _e_distplot)

    gravity_zone_results = {}
    for scenario_tag, result in diagnostic_results.items():
        tf_full, _, _ = _rebuild_full_trip_flows(
            result,
            fallback_inner_gdf=result.get("inner_gdf_plot", inner_gdf),
            ticks_per_day=TICKS_PER_DAY,
        )
        gravity_zone_results[scenario_tag] = _merge_supply_inbound(result, tf_full)

    gravity_prefix = os.path.join(CITY_OUT, f"fig_{CITY_SLUG}")
    _save_gravity_response_plots(gravity_zone_results, gravity_prefix)
    _save_exposure_source_plots(diagnostic_results, gravity_prefix)

    facility_burden_rows = []
    for scenario_tag, result in diagnostic_results.items():
        facility_burden_rows.append({"scenario": scenario_tag, **_facility_visit_burden_from_result(result, ticks=RUN_TICKS, ticks_per_day=TICKS_PER_DAY)})
    facility_burden_df = pd.DataFrame(facility_burden_rows)
    _save_facility_burden_plots(facility_burden_df, gravity_prefix)

    seed_scenarios = {
        "baseline": None,
        "intervB_mobilityreduce": {"type": "mobility_reduce", "rho_mean": 0.05, "gamma_mean": 1.15, "max_radius_m": 2500, "preserve_baseline_epi": True},
        "intervD_destinations_only": {
            "type": "area_redistribute",
            "scenario_tag": "intervD_destinations_only",
            "decentralisation_mode": "destinations_only",
            "attractiveness_mode": "centrality",
            "eta": 2.00,
            "beta": 0.15,
            "k_geom": 2.40,
            "delta_dest": 0.80,
            "delta_fac": 0.80,
            "lambda_blend": 0.45,
            "use_secondary_score_ranking": False,
            "alpha_fac": 0.80,
            "beta_fac": 0.35,
            "preserve_baseline_epi": True,
        },
        "intervD_facilities_only": {
            "type": "area_redistribute",
            "scenario_tag": "intervD_facilities_only",
            "decentralisation_mode": "sanitation_only",
            "attractiveness_mode": "centrality",
            "eta": 2.00,
            "beta": 0.15,
            "k_geom": 2.40,
            "delta_dest": 0.80,
            "delta_fac": 0.80,
            "lambda_blend": 0.45,
            "use_secondary_score_ranking": False,
            "alpha_fac": 0.80,
            "beta_fac": 0.35,
            "preserve_baseline_epi": True,
        },
        "intervD_area_redistribute": {
            "type": "area_redistribute",
            "scenario_tag": "intervD_area_redistribute",
            "decentralisation_mode": "all",
            "attractiveness_mode": "centrality",
            "eta": 2.00,
            "beta": 0.15,
            "k_geom": 2.40,
            "delta_dest": 0.80,
            "delta_fac": 0.80,
            "lambda_blend": 0.45,
            "use_secondary_score_ranking": False,
            "alpha_fac": 0.80,
            "beta_fac": 0.35,
            "preserve_baseline_epi": True,
        },
    }
    seed_values = [RUN_SEED, RUN_SEED + 1, RUN_SEED + 2]
    seed_df = _run_seed_variability(seed_values=seed_values, ticks=RUN_TICKS, scenarios=seed_scenarios)
    _save_seed_variability_plots(seed_df, gravity_prefix)

    print(f"[Trip diagnostics] saved summary → {trip_diag_summary_path}")
    print(f"[Trip diagnostics] saved metrics figure → {metrics_png}")
    print(f"[Trip diagnostics] saved O-D heatmaps → {heatmap_png}")

# ---- 125-run sensitivity analysis for Intervention D (k_geom × delta_dest × delta_fac) ----
D_SENSITIVITY_RESULTS = None
if RUN_D_SENSITIVITY:
    D_SENSITIVITY_RESULTS = run_intervention_d_sensitivity_analysis(
        ticks=RUN_TICKS,
        seed=RUN_SEED,
        ticks_per_day=TICKS_PER_DAY,
    )

# Sobol execution moved to the very end so all old plots/CSVs are produced first.

# -------- EXTRA: spatial hotspot KDE row AFTER interventions (A and B) --------
# This reuses the exact Cell 48 "Gaussian field + contourf + global linear scale" style,
# but swaps `base` for the intervention result dict (`res_A` / `res_B`).
def save_spatial_row_global_bright_from_result(sim_result: dict, scenario_tag: str):
    """
    Save the Cell-48 style spatial row (Initial / Peak / Final) for the provided sim_result.
    Writes into CITY_OUT alongside other saved figures.
    """
    import os
    import numpy as np
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as ctx
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap, ScalarMappable
    from scipy.spatial import cKDTree

    # knobs (match Cell 48)
    CMAP_NAME = "turbo"
    GRID_SPACING = 150.0
    SIGMA_M = 900.0
    CUTOFF_SIGMA = 3.5
    CONTOUR_LEVELS = 14
    EDGE_COLOR_OUTER = "#111111"
    EDGE_COLOR_INNER = "#ffffff"
    EDGE_W_OUTER = 1.2
    EDGE_W_INNER = 0.9
    WATER_BLUE = "#7EAAD9"
    WATER_RED = "#e53935"

    if sim_result is None or not isinstance(sim_result, dict):
        print(f"[Hotspot KDE] No sim_result for {scenario_tag}; skipping.")
        return
    if len(sim_result.get("snapshots", [])) == 0:
        print(f"[Hotspot KDE] No snapshots for {scenario_tag}; skipping.")
        return
    if sim_result.get("state_history", None) is None:
        print(f"[Hotspot KDE] No state_history for {scenario_tag}; skipping.")
        return

    # --- helpers (match Cell 48) ---
    def _to3857(gdf):
        if gdf is None:
            return None
        return gdf if str(getattr(gdf, "crs", None)) == "EPSG:3857" else gdf.to_crs("EPSG:3857")

    def _union(gdf):
        if gdf is None or gdf.empty:
            return None
        try:
            return gdf.union_all()
        except Exception:
            return gdf.unary_union

    def _contam_lookup(geom):
        rec = None
        if 'water_contamination_by_wkb' in globals():
            try:
                rec = water_contamination_by_wkb.get(geom.wkb)
            except Exception:
                rec = None
        if rec is None and 'water_contamination' in globals():
            rec = water_contamination.get(id(geom))
        return bool(rec and rec.get("contaminated", False))

    # --- layers (same globals as Cell 48) ---
    city_circle_gdf = None
    if 'boundaries_by_radius' in globals() and 'R_GLOBAL_M' in globals() and int(R_GLOBAL_M) in boundaries_by_radius:
        city_circle_gdf = _to3857(boundaries_by_radius[int(R_GLOBAL_M)])
    elif 'gdf_bound_3857' in globals():
        city_circle_gdf = _to3857(gdf_bound_3857)
    if city_circle_gdf is None or city_circle_gdf.empty:
        print(f"[Hotspot KDE] Missing city boundary/circle for {scenario_tag}; skipping.")
        return

    circle_union = _union(city_circle_gdf)

    outer_src = sim_result.get("outer_gdf_plot", None)
    inner_src = sim_result.get("inner_gdf_plot", None)

    if outer_src is None:
        outer_src = outer_gdf if ('outer_gdf' in globals()) else None
    if inner_src is None:
        inner_src = inner_gdf if ('inner_gdf' in globals()) else None

    outer_lines_raw = _to3857(outer_src) if outer_src is not None and (not getattr(outer_src, "empty", False)) else None
    inner_lines = _to3857(inner_src) if inner_src is not None and (not getattr(inner_src, "empty", False)) else None
    outer_lines = _build_cell_boundaries_from_inner_outer(inner_lines, outer_lines_raw) if outer_lines_raw is not None else None

    water_map = _to3857(water_gdf).copy() if 'water_gdf' in globals() else None
    if water_map is not None and not water_map.empty:
        water_map["contaminated"] = [_contam_lookup(g) for g in water_map.geometry]

    # --- ticks & titles (swap base -> sim_result) ---
    L = len(sim_result["snapshots"])
    I_series = sim_result["state_history"]["I"].values[:L]
    t_early = 0
    t_peak  = int(np.argmax(I_series)) if len(I_series) else 0
    t_late  = L - 1

    def _tick_to_day(t):
        tpd = globals().get("TICKS_PER_DAY", 24)
        return int(t // tpd) + 1

    day_initial = 1
    day_peak    = _tick_to_day(t_peak)
    day_final   = _tick_to_day(t_late)
    same_day_peak_final = (day_final == day_peak)
    titles = [
        f"Initial (Day {day_initial})",
        f"Peak (Day {day_peak})",
        f"Final (Day {day_final})" + (" — same as Peak" if same_day_peak_final else "")
    ]

    # --- grid & kernel (same as Cell 48) ---
    minx, miny, maxx, maxy = city_circle_gdf.total_bounds
    gx = np.arange(minx, maxx + GRID_SPACING, GRID_SPACING)
    gy = np.arange(miny, maxy + GRID_SPACING, GRID_SPACING)
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

    inside_mask = gpd.GeoSeries(
        gpd.points_from_xy(grid_pts[:, 0], grid_pts[:, 1]), crs="EPSG:3857"
    ).within(circle_union)
    mask2d = inside_mask.values.reshape(YY.shape)

    def _gaussian_field_for_tick(tick):
        snap = sim_result["snapshots"][tick]
        infected = snap[snap["state"].isin(["E", "I"])]  # EXPOSED + INFECTIOUS (same as Cell 48)

        Z = np.full(XX.shape, np.nan, dtype=float)
        if len(infected) == 0:
            Z[mask2d] = 0.0
            return Z

        # IMPORTANT: use the exact geometry column used in Cell 48
        pts = np.column_stack([
            infected["current_location"].x.values,
            infected["current_location"].y.values
        ])

        # Keep only sources within the circle (speed)
        gs = gpd.GeoSeries(gpd.points_from_xy(pts[:, 0], pts[:, 1]), crs="EPSG:3857")
        sel = gs.within(circle_union).values
        pts = pts[sel] if sel.any() else pts
        if pts.shape[0] == 0:
            Z[mask2d] = 0.0
            return Z

        tree = cKDTree(pts)
        inside_xy = grid_pts[inside_mask.values]
        rmax = CUTOFF_SIGMA * SIGMA_M
        B = 20000
        out_vals = np.zeros(inside_xy.shape[0], dtype=float)
        inv2s2 = 1.0 / (2.0 * SIGMA_M * SIGMA_M)

        for lo in range(0, inside_xy.shape[0], B):
            hi = min(lo + B, inside_xy.shape[0])
            qs = inside_xy[lo:hi]
            neigh_lists = tree.query_ball_point(qs, r=rmax)
            for j, idxs in enumerate(neigh_lists):
                if not idxs:
                    continue
                neigh = pts[idxs]
                dx = neigh[:, 0] - qs[j, 0]
                dy = neigh[:, 1] - qs[j, 1]
                d2 = dx * dx + dy * dy
                out_vals[lo + j] = np.exp(-d2 * inv2s2).sum()

        Zflat = np.full(grid_pts.shape[0], np.nan, dtype=float)
        Zflat[inside_mask.values] = out_vals
        return Zflat.reshape(YY.shape)

    Z_early = _gaussian_field_for_tick(t_early)
    Z_peak  = _gaussian_field_for_tick(t_peak)
    Z_late  = Z_peak.copy() if same_day_peak_final else _gaussian_field_for_tick(t_late)

    # --- global linear normalization (robust if all zeros) ---
    all_vals = np.concatenate([
        Z_early[~np.isnan(Z_early)].ravel(),
        Z_peak[~np.isnan(Z_peak)].ravel(),
        Z_late[~np.isnan(Z_late)].ravel()
    ])
    pos_vals = all_vals[all_vals > 0]

    if pos_vals.size == 0:
        vmin_global = -1e-9  # tiny negative to ensure contourf fills 0-valued fields
        vmax_global = 1.0
    else:
        vmin_global = 0.0
        vmax_global = float(np.max(pos_vals))
        if not np.isfinite(vmax_global) or vmax_global <= 0:
            vmax_global = 1.0

    norm_global = Normalize(vmin=vmin_global, vmax=vmax_global)
    levels = np.linspace(norm_global.vmin, norm_global.vmax, CONTOUR_LEVELS)

    cmap = get_cmap(CMAP_NAME)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw=dict(wspace=0.08))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Exposure + infectious density (states E+I): {scenario_tag}", fontsize=15, fontweight="bold")

    def _draw_panel(ax, Z, title):
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857", zorder=0)
        except Exception:
            pass

        ax.contourf(
            XX, YY, Z,
            levels=levels,
            cmap=cmap,
            norm=norm_global,
            zorder=2,
            antialiased=True
        )

        if outer_lines is not None and not outer_lines.empty:
            outer_lines.boundary.plot(ax=ax, color=EDGE_COLOR_OUTER, linewidth=EDGE_W_OUTER, alpha=1.0, zorder=4)
        if inner_lines is not None and not inner_lines.empty:
            inner_lines.boundary.plot(ax=ax, color=EDGE_COLOR_INNER, linewidth=EDGE_W_INNER, alpha=1.0, zorder=5)

        if water_map is not None and not water_map.empty:
            water_map.plot(
                ax=ax,
                color=water_map["contaminated"].map({False: WATER_BLUE, True: WATER_RED}),
                edgecolor="black",
                linewidth=0.35,
                alpha=0.9,
                zorder=3,
            )

        city_circle_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=6)

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title(title, fontsize=13, fontweight="bold", color="black", pad=8)
        ax.axis("off")
        ax.set_facecolor("white")

    for ax, title, Z in zip(axs, titles, [Z_early, Z_peak, Z_late]):
        _draw_panel(ax, Z, title)

    sm = ScalarMappable(norm=norm_global, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), location='right', fraction=0.03, pad=0.02)
    cbar.set_label("E+I density (exposed + infectious agents)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Save (next to existing city plots)
    assert 'CITY_SLUG' in globals() and 'CITY_OUT' in globals(), "Run R0a first."
    out_png = os.path.join(CITY_OUT, f"{CITY_SLUG}_{scenario_tag}_fig2_spatial_row_global_bright.png")
    try:
        plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Hotspot KDE] Saved {out_png}")
    finally:
        plt.close(fig)

    # Panel-normalized version for this scenario: each Initial / Peak / Final
    # panel is scaled to its own maximum to show hotspot concentration shape.
    def _normalize_field_panel_local(Z):
        Zn = np.array(Z, dtype=float, copy=True)
        finite = np.isfinite(Zn)
        positive = Zn[finite & (Zn > 0)]
        if positive.size == 0:
            Zn[finite] = 0.0
            return Zn
        Zn[finite] = Zn[finite] / float(np.nanmax(positive))
        return Zn

    Z_early_norm = _normalize_field_panel_local(Z_early)
    Z_peak_norm = _normalize_field_panel_local(Z_peak)
    Z_late_norm = _normalize_field_panel_local(Z_late)
    norm_panel = Normalize(vmin=0.0, vmax=1.0)
    levels_panel = np.linspace(0.0, 1.0, CONTOUR_LEVELS)
    fig_norm, axs_norm = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw=dict(wspace=0.08))
    fig_norm.patch.set_facecolor("white")
    fig_norm.suptitle(f"Exposure + infectious density (E+I), panel-normalized: {scenario_tag}", fontsize=15, fontweight="bold")

    def _draw_panel_norm(ax, Z, title):
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857", zorder=0)
        except Exception:
            pass
        ax.contourf(XX, YY, Z, levels=levels_panel, cmap=cmap, norm=norm_panel, zorder=2, antialiased=True)
        if outer_lines is not None and not outer_lines.empty:
            outer_lines.boundary.plot(ax=ax, color=EDGE_COLOR_OUTER, linewidth=EDGE_W_OUTER, alpha=1.0, zorder=4)
        if inner_lines is not None and not inner_lines.empty:
            inner_lines.boundary.plot(ax=ax, color=EDGE_COLOR_INNER, linewidth=EDGE_W_INNER, alpha=1.0, zorder=5)
        if water_map is not None and not water_map.empty:
            water_map.plot(ax=ax, color=water_map["contaminated"].map({False: WATER_BLUE, True: WATER_RED}), edgecolor="black", linewidth=0.35, alpha=0.9, zorder=3)
        city_circle_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=6)
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        ax.set_title(title, fontsize=13, fontweight="bold", color="black", pad=8)
        ax.axis("off"); ax.set_facecolor("white")

    for ax, title, Z in zip(axs_norm, titles, [Z_early_norm, Z_peak_norm, Z_late_norm]):
        _draw_panel_norm(ax, Z, title)
    sm_norm = ScalarMappable(norm=norm_panel, cmap=cmap)
    sm_norm.set_array([])
    cbar_norm = fig_norm.colorbar(sm_norm, ax=axs_norm.ravel().tolist(), location='right', fraction=0.03, pad=0.02)
    cbar_norm.set_label("Panel-normalized density (0–1)", fontsize=11)
    cbar_norm.ax.tick_params(labelsize=9)
    out_png_norm = os.path.join(CITY_OUT, f"{CITY_SLUG}_{scenario_tag}_fig2_spatial_row_panel_normalized.png")
    try:
        plt.savefig(out_png_norm, dpi=300, bbox_inches="tight", facecolor=fig_norm.get_facecolor())
        print(f"[Hotspot KDE normalized] Saved {out_png_norm}")
    finally:
        plt.close(fig_norm)

    # Matching hexbin version (same Initial / Peak / Final layout)
    HEXBIN_SIZE_M = GRID_SPACING * 4.0
    HEXBIN_GRIDSIZE_X = max(4, int(np.ceil((maxx - minx) / HEXBIN_SIZE_M)))
    HEXBIN_EXTENT = (minx, maxx, miny, maxy)

    def _hexbin_inputs_from_field(tick):
        snap = sim_result["snapshots"][tick]
        infected = snap[snap["state"].isin(["E", "I"])]
        if len(infected) == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float)
        xh = infected["current_location"].x.values.astype(float, copy=False)
        yh = infected["current_location"].y.values.astype(float, copy=False)
        gs = gpd.GeoSeries(gpd.points_from_xy(xh, yh), crs="EPSG:3857")
        sel = gs.within(circle_union).values
        if sel.any():
            xh = xh[sel]
            yh = yh[sel]
        return xh, yh, np.empty(xh.shape[0], dtype=float)

    x_early_hex, y_early_hex, c_early_hex = _hexbin_inputs_from_field(t_early)
    x_peak_hex,  y_peak_hex,  c_peak_hex  = _hexbin_inputs_from_field(t_peak)
    x_late_hex,  y_late_hex,  c_late_hex  = _hexbin_inputs_from_field(t_late)

    vmax_hex_global = 1.0
    _tmp_fig, _tmp_ax = plt.subplots()
    for xh, yh in zip([x_early_hex, x_peak_hex, x_late_hex], [y_early_hex, y_peak_hex, y_late_hex]):
        hb = _tmp_ax.hexbin(
            xh, yh,
            gridsize=HEXBIN_GRIDSIZE_X,
            extent=HEXBIN_EXTENT,
            mincnt=1,
        )
        if hb.get_array().size:
            _v = float(np.nanmax(hb.get_array()))
            if np.isfinite(_v) and _v > vmax_hex_global:
                vmax_hex_global = _v
        hb.remove()
    plt.close(_tmp_fig)

    norm_hex_global = Normalize(vmin=0.0, vmax=vmax_hex_global)

    fig_hex, axs_hex = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw=dict(wspace=0.08))
    fig_hex.patch.set_facecolor("white")

    def _draw_panel_hex(ax, xh, yh, ch, title):
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        def _path_from_polygon(poly):
            verts, codes = [], []
            xy = np.asarray(poly.exterior.coords)
            verts.extend(xy.tolist())
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
            for ring in poly.interiors:
                xy = np.asarray(ring.coords)
                verts.extend(xy.tolist())
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
            return Path(np.asarray(verts, dtype=float), codes)

        def _clip_patch(ax):
            geom = city_circle_gdf.geometry.union_all() if hasattr(city_circle_gdf.geometry, "union_all") else city_circle_gdf.geometry.unary_union
            if geom.geom_type == "Polygon":
                path = _path_from_polygon(geom)
            else:
                path = Path.make_compound_path(*[_path_from_polygon(g) for g in geom.geoms if not g.is_empty])
            return PathPatch(path, transform=ax.transData)

        hb = ax.hexbin(
            xh, yh,
            gridsize=HEXBIN_GRIDSIZE_X,
            extent=HEXBIN_EXTENT,
            mincnt=1,
            cmap=cmap,
            norm=norm_hex_global,
            linewidths=0,
            edgecolors="none",
            bins="log",
            alpha=0.95,
            zorder=2,
        )
        hb.set_clip_path(_clip_patch(ax))

        if outer_lines is not None and not outer_lines.empty:
            outer_lines.boundary.plot(ax=ax, color=EDGE_COLOR_OUTER,
                                      linewidth=EDGE_W_OUTER, alpha=1.0, zorder=4)
        if inner_lines is not None and not inner_lines.empty:
            inner_lines.boundary.plot(ax=ax, color=EDGE_COLOR_INNER,
                                      linewidth=EDGE_W_INNER, alpha=1.0, zorder=5)
        if water_map is not None and not water_map.empty:
            water_map.plot(
                ax=ax,
                color=water_map["contaminated"].map({False: WATER_BLUE, True: WATER_RED}),
                edgecolor="black",
                linewidth=0.35,
                alpha=0.9,
                zorder=3,
            )
        city_circle_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=6)

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title(title, fontsize=13, fontweight="bold", color="black", pad=8)
        ax.axis("off")
        ax.set_facecolor("white")

    for ax, title, xh, yh, ch in zip(
        axs_hex, titles,
        [x_early_hex, x_peak_hex, x_late_hex],
        [y_early_hex, y_peak_hex, y_late_hex],
        [c_early_hex, c_peak_hex, c_late_hex]
    ):
        _draw_panel_hex(ax, xh, yh, ch, title)

    sm_hex = ScalarMappable(norm=norm_hex_global, cmap=cmap)
    sm_hex.set_array([])
    cbar_hex = fig_hex.colorbar(sm_hex, ax=axs_hex.ravel().tolist(), location='right', fraction=0.03, pad=0.02)
    cbar_hex.set_label(f"Infected count per hexagon", fontsize=11)
    cbar_hex.ax.tick_params(labelsize=9)

    out_png_hex = os.path.join(CITY_OUT, f"{CITY_SLUG}_{scenario_tag}_fig2_spatial_row_global_bright_hexbin.png")
    try:
        plt.savefig(out_png_hex, dpi=300, bbox_inches="tight", facecolor=fig_hex.get_facecolor())
        print(f"[Hotspot Hexbin] Saved {out_png_hex}")
    finally:
        plt.close(fig_hex)


def save_peak_spatial_comparison_all_scenarios(scenario_results: dict, out_tag: str = "core_peak_all_scenarios"):
    """Save one peak-tick KDE panel per scenario with a single shared colorbar."""
    import os
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as ctx
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap, ScalarMappable
    from scipy.spatial import cKDTree

    scenario_results = {k: v for k, v in scenario_results.items() if isinstance(v, dict) and len(v.get("snapshots", [])) > 0}
    if not scenario_results:
        print("[Peak comparison KDE] No scenario snapshots available; skipping.")
        return

    CMAP_NAME = "turbo"
    GRID_SPACING = 150.0
    SIGMA_M = 900.0
    CUTOFF_SIGMA = 3.5
    CONTOUR_LEVELS = 14
    EDGE_COLOR_OUTER = "#111111"
    EDGE_COLOR_INNER = "#ffffff"
    EDGE_W_OUTER = 1.2
    EDGE_W_INNER = 0.9
    WATER_BLUE = "#7EAAD9"
    WATER_RED = "#e53935"

    def _to3857_local(gdf):
        if gdf is None:
            return None
        return gdf if str(getattr(gdf, "crs", None)) == "EPSG:3857" else gdf.to_crs("EPSG:3857")

    def _union_local(gdf):
        try:
            return gdf.union_all()
        except Exception:
            return gdf.unary_union

    city_gdf = None
    if 'boundaries_by_radius' in globals() and 'R_GLOBAL_M' in globals() and int(R_GLOBAL_M) in boundaries_by_radius:
        city_gdf = _to3857_local(boundaries_by_radius[int(R_GLOBAL_M)])
    elif 'gdf_bound_3857' in globals():
        city_gdf = _to3857_local(gdf_bound_3857)
    if city_gdf is None or city_gdf.empty:
        print("[Peak comparison KDE] Missing city boundary; skipping.")
        return
    circle = _union_local(city_gdf)
    minx, miny, maxx, maxy = city_gdf.total_bounds
    gx = np.arange(minx, maxx + GRID_SPACING, GRID_SPACING)
    gy = np.arange(miny, maxy + GRID_SPACING, GRID_SPACING)
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    inside_mask = gpd.GeoSeries(gpd.points_from_xy(grid_pts[:, 0], grid_pts[:, 1]), crs="EPSG:3857").within(circle)
    mask2d = inside_mask.values.reshape(YY.shape)
    inside_xy = grid_pts[inside_mask.values]

    water_map_local = _to3857_local(water_gdf).copy() if 'water_gdf' in globals() else None
    if water_map_local is not None and not water_map_local.empty:
        water_map_local["contaminated"] = [_contam_lookup(g) for g in water_map_local.geometry]

    def _field_for_result_at_peak(res):
        L = len(res.get("snapshots", []))
        hist = res.get("state_history", pd.DataFrame())
        I_series = hist["I"].values[:L] if isinstance(hist, pd.DataFrame) and "I" in hist.columns else np.zeros(L, dtype=float)
        t_peak = int(np.argmax(I_series)) if len(I_series) else 0
        snap = res["snapshots"][t_peak]
        infected = snap[snap["state"].isin(["E", "I"])]
        Z = np.full(XX.shape, np.nan, dtype=float)
        if len(infected) == 0:
            Z[mask2d] = 0.0
            return Z, t_peak, 0
        pts = np.column_stack([infected["current_location"].x.values, infected["current_location"].y.values])
        sel = gpd.GeoSeries(gpd.points_from_xy(pts[:, 0], pts[:, 1]), crs="EPSG:3857").within(circle).values
        pts = pts[sel] if sel.any() else pts
        if pts.shape[0] == 0:
            Z[mask2d] = 0.0
            return Z, t_peak, 0
        tree = cKDTree(pts)
        rmax = CUTOFF_SIGMA * SIGMA_M
        inv2s2 = 1.0 / (2.0 * SIGMA_M * SIGMA_M)
        out_vals = np.zeros(inside_xy.shape[0], dtype=float)
        B = 20000
        for lo in range(0, inside_xy.shape[0], B):
            hi = min(lo + B, inside_xy.shape[0])
            qs = inside_xy[lo:hi]
            neigh_lists = tree.query_ball_point(qs, r=rmax)
            for j, idxs in enumerate(neigh_lists):
                if not idxs:
                    continue
                neigh = pts[idxs]
                dx = neigh[:, 0] - qs[j, 0]
                dy = neigh[:, 1] - qs[j, 1]
                out_vals[lo + j] = np.exp(-(dx * dx + dy * dy) * inv2s2).sum()
        Zflat = np.full(grid_pts.shape[0], np.nan, dtype=float)
        Zflat[inside_mask.values] = out_vals
        return Zflat.reshape(YY.shape), t_peak, int(len(infected))

    fields = {}
    meta = {}
    for key, res in scenario_results.items():
        Z, t_peak, n_inf = _field_for_result_at_peak(res)
        fields[key] = Z
        meta[key] = {"tick": int(t_peak), "day": int(t_peak // globals().get("TICKS_PER_DAY", 24)) + 1, "n_EI": int(n_inf)}

    all_vals = np.concatenate([Z[np.isfinite(Z)].ravel() for Z in fields.values()])
    pos_vals = all_vals[all_vals > 0]
    vmax = float(np.nanmax(pos_vals)) if pos_vals.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    levels = np.linspace(0.0, vmax, CONTOUR_LEVELS)
    cmap = get_cmap(CMAP_NAME)

    n = len(fields)
    fig, axs = plt.subplots(1, n, figsize=(5.0 * n, 5.8), gridspec_kw=dict(wspace=0.06))
    axs = np.atleast_1d(axs).ravel()
    fig.patch.set_facecolor("white")
    fig.suptitle("Exposure + infectious density at peak tick: baseline vs current interventions", fontsize=15, fontweight="bold")

    for ax, (key, Z) in zip(axs, fields.items()):
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857", zorder=0)
        except Exception:
            pass
        ax.contourf(XX, YY, Z, levels=levels, cmap=cmap, norm=norm, zorder=2, antialiased=True)
        res = scenario_results[key]
        outer_src = res.get("outer_gdf_plot", None)
        if outer_src is None:
            outer_src = outer_gdf if 'outer_gdf' in globals() else None
        inner_src = res.get("inner_gdf_plot", None)
        if inner_src is None:
            inner_src = inner_gdf if 'inner_gdf' in globals() else None
        outer_raw = _to3857_local(outer_src) if outer_src is not None and not getattr(outer_src, "empty", False) else None
        inner_lines_local = _to3857_local(inner_src) if inner_src is not None and not getattr(inner_src, "empty", False) else None
        outer_lines_local = _build_cell_boundaries_from_inner_outer(inner_lines_local, outer_raw) if outer_raw is not None else None
        if outer_lines_local is not None and not outer_lines_local.empty:
            outer_lines_local.boundary.plot(ax=ax, color=EDGE_COLOR_OUTER, linewidth=EDGE_W_OUTER, alpha=1.0, zorder=4)
        if inner_lines_local is not None and not inner_lines_local.empty:
            inner_lines_local.boundary.plot(ax=ax, color=EDGE_COLOR_INNER, linewidth=EDGE_W_INNER, alpha=1.0, zorder=5)
        if water_map_local is not None and not water_map_local.empty:
            water_map_local.plot(ax=ax, color=water_map_local["contaminated"].map({False: WATER_BLUE, True: WATER_RED}), edgecolor="black", linewidth=0.35, alpha=0.9, zorder=3)
        city_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=6)
        m = meta[key]
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        ax.set_title(f"{_pretty_tripdiag_label(key)}\nPeak day {m['day']}, tick {m['tick']}; N E+I={m['n_EI']}", fontsize=11, fontweight="bold")
        ax.axis("off"); ax.set_facecolor("white")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), location="right", fraction=0.03, pad=0.02)
    cbar.set_label("E+I density (exposed + infectious agents)", fontsize=11)
    out_png = os.path.join(CITY_OUT, f"{CITY_SLUG}_{out_tag}.png")
    out_csv = os.path.join(CITY_OUT, f"{CITY_SLUG}_{out_tag}_metadata.csv")
    pd.DataFrame([{"scenario": k, **v} for k, v in meta.items()]).to_csv(out_csv, index=False)
    try:
        plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Peak comparison KDE] Saved {out_png}")
    finally:
        plt.close(fig)

# Actually save extra hotspot rows for interventions (keep baseline plot unchanged)
try:
    if 'res_A' in globals() and res_A is not None:
        save_spatial_row_global_bright_from_result(res_A, "intervA_decentralise")
    if 'res_B' in globals() and res_B is not None:
        save_spatial_row_global_bright_from_result(res_B, "intervB_mobilityreduce")
    if 'res_C' in globals() and res_C is not None:
        save_spatial_row_global_bright_from_result(res_C, "intervC_combined")
    if 'res_D_dest' in globals() and res_D_dest is not None:
        save_spatial_row_global_bright_from_result(res_D_dest, "intervD_destinations_only")
    if 'res_D_fac' in globals() and res_D_fac is not None:
        save_spatial_row_global_bright_from_result(res_D_fac, "intervD_facilities_only")
    if 'res_D' in globals() and res_D is not None:
        save_spatial_row_global_bright_from_result(res_D, "intervD_area_redistribute")
    try:
        peak_compare_results = {
            "baseline": base,
            "intervB_mobilityreduce": res_B,
            "intervD_destinations_only": globals().get("res_D_dest", None),
            "intervD_facilities_only": globals().get("res_D_fac", None),
            "intervD_area_redistribute": res_D,
        }
        save_peak_spatial_comparison_all_scenarios(peak_compare_results)
    except Exception as _e_peak_compare:
        print(f"[Peak comparison KDE] failed: {_e_peak_compare}")
except Exception as e:
    print(f"[Hotspot KDE] failed after interventions: {e}")


# ===================== Notebook cell 53 =====================
# R8 — FIGURES (per-city)
# Produces:
#   Fig.1: baseline vs Interventions A/B/C/D (daily incidence across all simulated days)
#   Fig.3: Infrastructure exposure (if *_visits.csv, *_contam.csv, *_weeklyexp.csv exist)
# Note: Old intervention figures (restrict/sanitation/surveillance) removed.

import os, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt

assert 'CITY_SLUG' in globals() and 'SHARED_OUT' in globals() and 'CITY_OUT' in globals(), "Run R0a/R0b first."
os.makedirs(SHARED_OUT, exist_ok=True)
os.makedirs(CITY_OUT, exist_ok=True)

def _file_nonempty(p):
    try:
        return os.path.isfile(p) and os.path.getsize(p) > 0
    except Exception:
        return False

def _read_csv_safe(path, required_cols):
    if not path or not os.path.exists(path) or not _file_nonempty(path):
        return None
    try:
        df = pd.read_csv(path)
        if not set(required_cols).issubset(df.columns):
            return None
        return df
    except Exception as e:
        print(f"[R8] Skipping unreadable CSV '{os.path.basename(path)}': {e}")
        return None

def _to_day(df_tick, tpd=24, n_days=None):
    d = df_tick.copy()
    d["day"] = (d["tick"] // tpd) + 1
    daily = d.groupby("day")["inc"].sum()
    if n_days is None:
        if len(daily):
            n_days = int(daily.index.max())
        else:
            n_days = int(np.ceil(float(globals().get("RUN_TICKS", 0)) / max(float(tpd), 1.0)))
    n_days = max(int(n_days), 1)
    return daily.reindex(range(1, n_days + 1)).fillna(0.0)

TPD = globals().get("TICKS_PER_DAY", 24)
N_DAYS_PLOT = max(1, int(np.ceil(float(globals().get("RUN_TICKS", 0)) / max(float(TPD), 1.0))))

# ========= Fig. 1 — Epidemic trajectory (incidence %, full simulation days): baseline vs A/B/C/D =========
# Baseline (preferred): {CITY_SLUG}_incidence.csv from R4
base_tick = _read_csv_safe(os.path.join(SHARED_OUT, f"{CITY_SLUG}_incidence.csv"), required_cols=["tick","inc"])
# fallback baseline name used elsewhere
if base_tick is None:
    base_tick = _read_csv_safe(os.path.join(SHARED_OUT, f"{CITY_SLUG}_base_inc.csv"), required_cols=["tick","inc"])

# Intervention incidence files from R7 exports
intervention_incidence_files = {
    "Intervention B: mobility↓": f"{CITY_SLUG}_intervB_mobilityreduce_incidence.csv",
    "D1: activity only": f"{CITY_SLUG}_intervD_destinations_only_incidence.csv",
    "D2: taps + toilets only": f"{CITY_SLUG}_intervD_facilities_only_incidence.csv",
    "D3: all amenities": f"{CITY_SLUG}_intervD_area_redistribute_incidence.csv",
}
intervention_incidence_ticks = {
    label: _read_csv_safe(os.path.join(SHARED_OUT, fname), required_cols=["tick", "inc"])
    for label, fname in intervention_incidence_files.items()
}

if base_tick is not None:
    base_day = _to_day(base_tick, tpd=TPD, n_days=N_DAYS_PLOT)

    fig, ax = plt.subplots(figsize=(7.5,4.5), constrained_layout=True)
    ax.plot(base_day.index, base_day.values, color="black", label="Baseline", linewidth=2)

    for label, tick_df in intervention_incidence_ticks.items():
        if tick_df is not None:
            day_series = _to_day(tick_df, tpd=TPD, n_days=N_DAYS_PLOT)
            ax.plot(day_series.index, day_series.values, label=label, linewidth=2)
        else:
            print(f"[Fig.1] Note: missing {label} incidence CSV; skipping that line.")

    ax.set_title(f"{CITY_SLUG.replace('_',' ').title()} — Daily Incidence (%)")
    ax.set_xlabel("Day"); ax.set_ylabel("Daily Incidence (%)")
    ax.set_xlim(1, N_DAYS_PLOT)
    ax.set_xticks(list(range(1, N_DAYS_PLOT + 1)))
    ax.grid(alpha=.25)
    ax.legend(frameon=False)

    out_png = os.path.join(CITY_OUT, f"fig1_{CITY_SLUG}.png")
    plt.savefig(OUT_DIR /out_png, dpi=220);
    #plt.show()
    print(f"[Fig.1] Saved {out_png}")
else:
    print("[Fig.1] Skipped: missing baseline tick-incidence CSV for this city.")

# ========= Fig. 3 — Infrastructure exposure (single city) =========
# (kept as-is; not intervention-specific)
vis = _read_csv_safe(os.path.join(SHARED_OUT, f"{CITY_SLUG}_visits.csv"), required_cols=["node_id","visits"])
con = _read_csv_safe(os.path.join(SHARED_OUT, f"{CITY_SLUG}_contam.csv"), required_cols=["node_id","contaminated"])
wex = _read_csv_safe(os.path.join(SHARED_OUT, f"{CITY_SLUG}_weeklyexp.csv"), required_cols=["agent","week","count"])

if all(df is not None for df in [vis, con, wex]):
    df = vis.merge(con, on="node_id", how="left").fillna({'visits':0,'contaminated':0})
    fig, axs = plt.subplots(1,3, figsize=(14,4), constrained_layout=True)

    # (a) Node visitation vs contamination (log-log +1)
    axs[0].scatter(
        df["visits"]+1,
        pd.to_numeric(df.get("feces_load", pd.Series([0]*len(df))), errors='coerce').fillna(0)+1,
        s=18, alpha=.5
    )
    axs[0].set_xscale('log'); axs[0].set_yscale('log')
    axs[0].set_xlabel("Visits (+1, log)"); axs[0].set_ylabel("Feces load (+1, log)")
    axs[0].set_title("Node visitation vs contamination")

    # (b) Contaminated share by visit decile
    d = df.copy().sort_values("visits").reset_index(drop=True)
    d["rk"] = np.arange(1, len(d)+1)
    if len(d) >= 10:
        d["dec"] = pd.qcut(d['rk'], q=10, labels=False, duplicates='drop')
    else:
        d["dec"] = 0
    dec = (
        d.groupby('dec')["contaminated"]
        .apply(lambda s: pd.to_numeric(s, errors='coerce').fillna(0).mean())
        * 100
    )
    axs[1].plot(dec.index, dec.values, marker='o')
    axs[1].set_xlabel("Visit decile (low→high)")
    axs[1].set_ylabel("Contaminated share (%)")
    axs[1].set_title("Contaminated share by visits")

    # (c) Exposure distribution (weekly S→E)
    bins = [0,1,2,3,5,8,13,21,34]
    axs[2].hist(wex["count"], bins=bins, alpha=.7, density=True)
    axs[2].set_xlabel("Weekly exposure events")
    axs[2].set_ylabel("Density")
    axs[2].set_title("Exposure distribution")

    out_png = os.path.join(CITY_OUT, f"fig3_{CITY_SLUG}.png")
    plt.savefig(OUT_DIR /out_png, dpi=220);
    #plt.show()
    print(f"[Fig.3] Saved {out_png}")
else:
    print("[Fig.3] Skipped: need *_visits.csv, *_contam.csv, *_weeklyexp.csv for this city.")

print("[R8] Done. Old-intervention figures removed (restrict/sanitation/surveillance).")

# ===================== Notebook cell 54 =====================
# R9 — TABLE 1 (per-city metrics; incidence %; peak time by I-peak)
# (CHANGED) Per-city only (no mono/poly). Uses CITY_SLUG files saved earlier.
# (CHANGED) Outputs a per-city CSV (and optional LaTeX) named with CITY_SLUG into CITY_OUT.

import os, json, numpy as np, pandas as pd

# Ensure directories from R0a exist
assert 'CITY_SLUG' in globals() and 'CITY_OUT' in globals() and 'SHARED_OUT' in globals(), "Run R0a first."
os.makedirs(SHARED_OUT, exist_ok=True)
os.makedirs(CITY_OUT, exist_ok=True)

def _read_state_history(tag):
    """Load state history exported in R4 for this city (columns S,E,I,R)."""
    p = os.path.join(SHARED_OUT, f"{tag}_state_history.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing state history for {tag}: {p}")
    df = pd.read_csv(p)
    for col in ["S","E","I","R"]:
        if col not in df.columns:
            raise ValueError(f"{p} missing '{col}' column")
    return df

def _incidence_percent_from_hist(hist_df):
    """
    Incidence % per tick = 100 * new S→E infections / N.
    E→I is disease progression and is not counted as a new case here.
    """
    N = int(hist_df.iloc[0][["S","E","I","R"]].sum())
    S = hist_df["S"].to_numpy()
    dSE = np.maximum(0, -np.diff(np.r_[S[0], S]))  # S decrease = new exposures
    inc_pct = 100.0 * dSE / max(N, 1)
    return inc_pct

def _load_spread_km(tag):
    """
    Prefer metrics.json if present, else spread.json saved in R4.
    Supports keys {"spatial_spread_radius_km"} or {"radius_km"}.
    """
    for p in [os.path.join(SHARED_OUT, f"{tag}_metrics.json"),
              os.path.join(SHARED_OUT, f"{tag}_spread.json")]:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    j = json.load(f)
                val = j.get("spatial_spread_radius_km", j.get("radius_km", None))
                if val is not None:
                    return float(val)
            except Exception:
                pass
    return float("nan")

def _row_for_city(tag):
    """Assemble one-row summary for this city."""
    hist = _read_state_history(tag)
    # time to peak by maximum I (tick index)
    I = hist["I"].to_numpy()
    t_peak = int(np.argmax(I)) if len(I) else 0
    # peak incidence % (max across ticks)
    inc_pct = _incidence_percent_from_hist(hist)
    peak_inc_pct = float(np.nanmax(inc_pct)) if inc_pct.size else 0.0
    # spread radius (km)
    spread_km = _load_spread_km(tag)
    # prefer human-readable city name if available
    city_label = globals().get("CITY_NAME", tag)
    return {
        "City": city_label,
        "Time to Peak (ticks)": t_peak,
        "Peak Incidence (%)": round(peak_inc_pct, 2),
        "Spatial Spread Radius (km)": (round(spread_km, 2) if np.isfinite(spread_km) else np.nan),
    }

# Build table for this city
row = _row_for_city(CITY_SLUG)
table = pd.DataFrame([row], columns=["City","Time to Peak (ticks)","Peak Incidence (%)","Spatial Spread Radius (km)"])

# Save per-city CSV (+ optional LaTeX with city-specific filename)
out_csv = os.path.join(CITY_OUT, f"{CITY_SLUG}_table_metrics.csv")
out_tex = os.path.join(CITY_OUT, f"{CITY_SLUG}_table_metrics.tex")
table.to_csv(out_csv, index=False)
with open(out_tex, "w") as f:
    f.write(
        table.to_latex(
            index=False, float_format="%.2f",
            caption="\\textbf{Outbreak indicators for the city.}",
            label=f"tab:metrics_{CITY_SLUG}"
        )
    )



print(f"[Table] Saved {out_csv} and {out_tex}")

# ===================== CO-OCCURRENCE ENSEMBLE ANALYSIS =====================
# Main paper analysis: baseline + D1 + D2 + D3 only. Mobility reduction remains
# available elsewhere as a sensitivity comparator. Sobol is forced off whenever
# RUN_COOCCURRENCE_ENSEMBLE=True.

from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from matplotlib.colors import Normalize

COOCC_OUT = Path(CITY_OUT) / "cooccurrence_ensemble"
COOCC_OUT.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = COOCC_OUT / "paper_figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# The folder contains individual assets used by the
# main study figures. Other diagnostic and supplementary outputs
# are written under paper_figures/.
FINAL_PAPER_MAIN_DIR = COOCC_OUT / "final_paper_main"
FINAL_PAPER_MAIN_DIR.mkdir(parents=True, exist_ok=True)


COOCC_SCENARIOS = {
    "baseline": None,
    "intervD_destinations_only": _base_intervention_d_dict("destinations_only", "intervD_destinations_only"),
    "intervD_facilities_only": _base_intervention_d_dict("sanitation_only", "intervD_facilities_only"),
    "intervD_area_redistribute": _base_intervention_d_dict("all", "intervD_area_redistribute"),
}
COOCC_LABELS = {
    "baseline": "Baseline clustered",
    "intervD_destinations_only": "D1 activity redistribution",
    "intervD_facilities_only": "D2 WASH redistribution",
    "intervD_area_redistribute": "D3 activity + WASH redistribution",
}
PAPER_SCENARIO_LABELS = dict(COOCC_LABELS)

# Study labels for final output panels. Keeping COOCC_LABELS
# consistent preserves scenario identity throughout outputs.
FINAL_PAPER_SCENARIO_LABELS = {
    "baseline": "Baseline",
    "intervD_destinations_only": "Activity-only",
    "intervD_facilities_only": "WASH-only",
    "intervD_area_redistribute": "Activity & WASH",
}

PAPER_FEATURE_LABELS = {
    "generic_activity": "Generic activity",
    "school": "Schools",
    "market": "Markets",
    "tap": "Water taps",
    "toilet": "Public toilets",
    "home": "Homes",
}
COOCC_ORDER = list(COOCC_SCENARIOS)
COOCC_FEATURE_ORDER = ["school", "market", "generic_activity", "tap", "toilet"]


def _safe_gini(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0.0, None)
    if x.size == 0 or float(x.sum()) <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1.0) / n)


def _share_summary(values):
    x = np.asarray(values, dtype=float)
    x = np.clip(np.nan_to_num(x, nan=0.0), 0.0, None)
    if x.size == 0 or x.sum() <= 0:
        return {"top1_share_pct": 0.0, "top2_share_pct": 0.0,
                "normalised_entropy": 0.0, "gini": 0.0}
    p = x / x.sum()
    ps = np.sort(p)[::-1]
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    max_ent = float(np.log(len(p))) if len(p) > 1 else 1.0
    return {
        "top1_share_pct": 100.0 * float(ps[0]),
        "top2_share_pct": 100.0 * float(ps[:2].sum()),
        "normalised_entropy": float(entropy / max_ent) if max_ent > 0 else 0.0,
        "gini": _safe_gini(x),
    }


def _coords_from_gdf(gdf):
    if gdf is None or getattr(gdf, "empty", True):
        return np.empty((0, 2), dtype=float)
    return np.column_stack((gdf.geometry.x.to_numpy(dtype=float), gdf.geometry.y.to_numpy(dtype=float)))


def _unique_home_gdf(initial_agents):
    if initial_agents is None or initial_agents.empty:
        return gpd.GeoDataFrame(columns=["household_id", "geometry"], geometry="geometry", crs="EPSG:3857")
    d = initial_agents.drop_duplicates("household_id").copy()
    d = d[np.isfinite(d["home_x"]) & np.isfinite(d["home_y"])]
    return gpd.GeoDataFrame(d[["household_id"]], geometry=gpd.points_from_xy(d.home_x, d.home_y), crs="EPSG:3857")


def _active_layer(result, feature_type):
    key = {
        "school": "schools_active", "market": "markets_active",
        "generic_activity": "generic_active", "tap": "taps_active",
        "toilet": "toilets_active"
    }[feature_type]
    g = result.get(key, None)
    if g is None:
        return gpd.GeoDataFrame(columns=["feature_id", "feature_type", "weight", "geometry"], geometry="geometry", crs="EPSG:3857")
    return g.copy()


def _nearest_pair_rows(result, scenario, seed, radii=(250.0, 500.0, 1000.0)):
    layers = {k: _active_layer(result, k) for k in COOCC_FEATURE_ORDER}
    layers["home"] = _unique_home_gdf(result.get("initial_agents", pd.DataFrame()))
    pairs = [
        ("school", "market"), ("school", "tap"), ("market", "tap"),
        ("school", "toilet"), ("market", "toilet"),
        ("home", "tap"), ("home", "toilet"),
    ]
    rows = []
    for a, b in pairs:
        A = _coords_from_gdf(layers[a]); B = _coords_from_gdf(layers[b])
        if len(A) == 0 or len(B) == 0:
            for r in radii:
                rows.append({"scenario": scenario, "seed": seed, "feature_A": a, "feature_B": b,
                             "radius_m": r, "mean_nearest_distance_m": np.nan,
                             "median_nearest_distance_m": np.nan, "pct_A_with_B_in_radius": np.nan,
                             "mean_B_count_in_radius": np.nan})
            continue
        tree = cKDTree(B)
        dist, _ = tree.query(A, k=1)
        for r in radii:
            counts = np.asarray([len(x) for x in tree.query_ball_point(A, r=r)], dtype=float)
            rows.append({
                "scenario": scenario, "seed": seed, "feature_A": a, "feature_B": b,
                "radius_m": float(r), "mean_nearest_distance_m": float(np.mean(dist)),
                "median_nearest_distance_m": float(np.median(dist)),
                "pct_A_with_B_in_radius": 100.0 * float(np.mean(counts > 0)),
                "mean_B_count_in_radius": float(np.mean(counts)),
            })
    return rows


def _assign_points_to_polys(coords, polys_gdf):
    if polys_gdf is None or getattr(polys_gdf, "empty", True) or len(coords) == 0:
        return np.full(len(coords), -1, dtype=int)
    polys = polys_gdf.reset_index(drop=True).copy()
    try:
        sidx = polys.sindex
    except Exception:
        sidx = None
    cents = np.column_stack((polys.geometry.centroid.x, polys.geometry.centroid.y))
    out = []
    for x, y in coords:
        pt = Point(float(x), float(y)); found = -1
        cand = list(sidx.intersection(pt.bounds)) if sidx is not None else range(len(polys))
        for i in cand:
            try:
                if polys.geometry.iloc[int(i)].contains(pt) or polys.geometry.iloc[int(i)].touches(pt):
                    found = int(i); break
            except Exception:
                pass
        if found < 0 and len(cents):
            found = int(np.argmin(np.hypot(cents[:, 0] - x, cents[:, 1] - y)))
        out.append(found)
    return np.asarray(out, dtype=int)


def _hub_distribution(values_coords, values_weights, hubs):
    n = 0 if hubs is None else len(hubs)
    if n == 0:
        return np.asarray([], dtype=float)
    ids = _assign_points_to_polys(values_coords, hubs)
    out = np.zeros(n, dtype=float)
    for i, w in zip(ids, values_weights):
        if 0 <= int(i) < n:
            out[int(i)] += float(w)
    return out


def _urban_form_rows(result, scenario, seed):
    hubs = result.get("inner_hubs_active", None)
    if hubs is None or getattr(hubs, "empty", True):
        return []
    rows = []
    layers = {}
    layers["hub_area"] = np.asarray(hubs.geometry.area, dtype=float)

    act_coords, act_weights = [], []
    for ft in ["generic_activity", "school", "market"]:
        g = _active_layer(result, ft)
        if not g.empty:
            act_coords.extend(_coords_from_gdf(g).tolist())
            if ft == "generic_activity" and "weight" in g.columns:
                act_weights.extend(pd.to_numeric(g["weight"], errors="coerce").fillna(0).tolist())
            else:
                act_weights.extend([1.0] * len(g))
    layers["activity_supply"] = _hub_distribution(np.asarray(act_coords), np.asarray(act_weights), hubs) if act_coords else np.zeros(len(hubs))

    wash_coords = []
    for ft in ["tap", "toilet"]:
        g = _active_layer(result, ft)
        if not g.empty:
            wash_coords.extend(_coords_from_gdf(g).tolist())
    layers["wash_supply"] = _hub_distribution(np.asarray(wash_coords), np.ones(len(wash_coords)), hubs) if wash_coords else np.zeros(len(hubs))

    visits = result.get("completed_visits", pd.DataFrame())
    if visits is not None and not visits.empty:
        vc = visits[["facility_x", "facility_y"]].to_numpy(dtype=float)
        layers["realised_visits"] = _hub_distribution(vc, np.ones(len(vc)), hubs)
    else:
        layers["realised_visits"] = np.zeros(len(hubs))

    for layer, vals in layers.items():
        summ = _share_summary(vals)
        rows.append({"scenario": scenario, "seed": seed, "layer": layer, **summ})
    return rows


def _amenity_attraction_rows(result, scenario, seed):
    visits = result.get("completed_visits", pd.DataFrame())
    N = max(int(result.get("N", 1)), 1)
    n_days = max(float(RUN_TICKS) / float(TICKS_PER_DAY), 1.0)
    rows = []
    for ft in COOCC_FEATURE_ORDER:
        d = visits[visits["facility_type"].eq(ft)].copy() if visits is not None and not visits.empty else pd.DataFrame()
        if d.empty:
            rows.append({"scenario": scenario, "seed": seed, "facility_type": ft,
                         "visits_per_agent_day": 0.0, "total_visits": 0, "unique_visitors": 0,
                         "mean_trip_distance_m": np.nan, "median_trip_distance_m": np.nan,
                         "mean_travel_time_ticks": np.nan, "median_travel_time_ticks": np.nan,
                         "visits_per_facility": 0.0, "top1_visit_share_pct": 0.0,
                         "top3_visit_share_pct": 0.0, "visit_gini": 0.0,
                         "exposed_visitors_by_day2_pct": np.nan,
                         "exposed_visitors_by_day3_pct": np.nan})
            continue
        counts = d.groupby("facility_id", dropna=False).size().to_numpy(dtype=float)
        _exp = result.get("exposures", pd.DataFrame()).copy()
        _init = result.get("initial_agents", pd.DataFrame()).rename(columns={"agent_id":"agent"}).copy()
        if _exp is not None and not _exp.empty and "agent" in _exp.columns:
            _first = _exp.sort_values("tick").drop_duplicates("agent", keep="first")[["agent","day"]].rename(columns={"day":"exposure_day"})
            _vis_agents = pd.DataFrame({"agent": sorted(set(pd.to_numeric(d["agent"], errors="coerce").dropna().astype(int)))})
            _vis_agents = _vis_agents.merge(_init[["agent","initial_state"]], on="agent", how="left").merge(_first, on="agent", how="left")
            _vis_agents = _vis_agents[_vis_agents["initial_state"].eq("S")]
            _early2 = 100.0*float(_vis_agents["exposure_day"].le(2).mean()) if len(_vis_agents) else np.nan
            _early3 = 100.0*float(_vis_agents["exposure_day"].le(3).mean()) if len(_vis_agents) else np.nan
        else:
            _early2 = np.nan; _early3 = np.nan
        rows.append({
            "scenario": scenario, "seed": seed, "facility_type": ft,
            "visits_per_agent_day": float(len(d) / (N * n_days)), "total_visits": int(len(d)),
            "unique_visitors": int(d["agent"].nunique()),
            "mean_trip_distance_m": float(pd.to_numeric(d.get("journey_distance_m", d["actual_trip_distance_m"]), errors="coerce").fillna(pd.to_numeric(d["actual_trip_distance_m"], errors="coerce")).mean()),
            "median_trip_distance_m": float(pd.to_numeric(d.get("journey_distance_m", d["actual_trip_distance_m"]), errors="coerce").fillna(pd.to_numeric(d["actual_trip_distance_m"], errors="coerce")).median()),
            "mean_travel_time_ticks": float(pd.to_numeric(d.get("travel_time_ticks", pd.Series(np.nan, index=d.index)), errors="coerce").mean()),
            "median_travel_time_ticks": float(pd.to_numeric(d.get("travel_time_ticks", pd.Series(np.nan, index=d.index)), errors="coerce").median()),
            "visits_per_facility": float(len(d) / max(d["facility_id"].nunique(), 1)),
            "top1_visit_share_pct": 100.0 * float(np.sort(counts)[-1:].sum() / counts.sum()),
            "top3_visit_share_pct": 100.0 * float(np.sort(counts)[-3:].sum() / counts.sum()),
            "visit_gini": _safe_gini(counts),
            "exposed_visitors_by_day2_pct": float(_early2) if np.isfinite(_early2) else np.nan,
            "exposed_visitors_by_day3_pct": float(_early3) if np.isfinite(_early3) else np.nan,
        })
    return rows


def _agent_covisitation_table(result, visits_override=None):
    """Return agent-level visit-combination flags.

    `visits_override` lets risk analyses use only visits observed before an
    exposure/outcome horizon.  Ordinary co-visitation plots still use the full
    completed diary, so no existing model mechanism is altered.
    """
    visits = visits_override if visits_override is not None else result.get("completed_visits", pd.DataFrame())
    init = result.get("initial_agents", pd.DataFrame()).copy()
    if init.empty:
        return pd.DataFrame()
    agents = init.rename(columns={"agent_id": "agent"})[["agent", "role", "initial_state"]].copy()
    flag_cols = [
        "visited_school", "visited_market", "visited_generic_activity", "visited_tap", "visited_toilet",
        "school_market_same_day", "school_tap_same_day", "market_tap_same_day",
        "activity_tap_same_day", "activity_toilet_same_day", "activity_wash_same_day",
        "activity_to_tap_within_6h",
    ]
    for c in flag_cols:
        agents[c] = False
    if visits is None or visits.empty:
        return agents
    visits = visits.copy()
    visits["agent"] = pd.to_numeric(visits["agent"], errors="coerce").astype("Int64")
    visits = visits.dropna(subset=["agent", "tick", "facility_type"])
    visits["agent"] = visits["agent"].astype(int)
    for ft in COOCC_FEATURE_ORDER:
        ids = set(visits.loc[visits["facility_type"].eq(ft), "agent"].astype(int))
        agents.loc[agents["agent"].isin(ids), f"visited_{ft}"] = True
    by_day = visits.groupby(["agent", "day"])["facility_type"].agg(lambda x: set(x)).reset_index()
    pair_map = {
        "school_market_same_day": lambda s: "school" in s and "market" in s,
        "school_tap_same_day": lambda s: "school" in s and "tap" in s,
        "market_tap_same_day": lambda s: "market" in s and "tap" in s,
        "activity_tap_same_day": lambda s: bool(s.intersection({"school","market","generic_activity"})) and "tap" in s,
        "activity_toilet_same_day": lambda s: bool(s.intersection({"school","market","generic_activity"})) and "toilet" in s,
        "activity_wash_same_day": lambda s: bool(s.intersection({"school","market","generic_activity"})) and bool(s.intersection({"tap","toilet"})),
    }
    for col, fn in pair_map.items():
        ids = set(by_day.loc[by_day["facility_type"].apply(fn), "agent"].astype(int))
        agents.loc[agents["agent"].isin(ids), col] = True
    seq_ids = set()
    for uid, d in visits.sort_values(["agent","tick"]).groupby("agent"):
        act_ticks = d.loc[d["facility_type"].isin(["school","market","generic_activity"]), "tick"].to_numpy(dtype=int)
        tap_ticks = d.loc[d["facility_type"].eq("tap"), "tick"].to_numpy(dtype=int)
        if len(act_ticks) and len(tap_ticks):
            for a_tick in act_ticks:
                if np.any((tap_ticks >= a_tick) & (tap_ticks - a_tick <= 6)):
                    seq_ids.add(int(uid)); break
    agents.loc[agents["agent"].isin(seq_ids), "activity_to_tap_within_6h"] = True
    return agents


def _contact_opportunity_rows(result, scenario, seed):
    """Co-presence opportunities at the same facility and tick.

    These are contact opportunities only; the current cholera model does not
    transmit directly between non-household agents at schools/markets.
    """
    visits = result.get("completed_visits", pd.DataFrame())
    N = max(int(result.get("N", 1)), 1)
    days = max(float(RUN_TICKS) / float(TICKS_PER_DAY), 1.0)
    rows = []
    for ft in COOCC_FEATURE_ORDER:
        d = visits[visits["facility_type"].eq(ft)].copy() if visits is not None and not visits.empty else pd.DataFrame()
        if d.empty:
            rows.append({"scenario":scenario,"seed":seed,"facility_type":ft,
                         "pct_visits_with_copresence":0.0,"mean_agents_per_occupied_event":0.0,
                         "contact_pairs_per_agent_day":0.0})
            continue
        occ = d.groupby(["tick","facility_id"], dropna=False)["agent"].nunique().astype(float)
        visit_weighted_with_other = float((occ[occ > 1].sum()) / max(occ.sum(), 1.0))
        pair_count = float(np.sum(occ * (occ - 1.0) / 2.0))
        rows.append({"scenario":scenario,"seed":seed,"facility_type":ft,
                     "pct_visits_with_copresence":100.0*visit_weighted_with_other,
                     "mean_agents_per_occupied_event":float(occ.mean()),
                     "contact_pairs_per_agent_day":pair_count/(N*days)})
    return rows


def _covisitation_rows(result, scenario, seed):
    a = _agent_covisitation_table(result)
    if a.empty:
        return []
    cols = ["school_market_same_day","school_tap_same_day","market_tap_same_day",
            "activity_tap_same_day","activity_toilet_same_day","activity_wash_same_day",
            "activity_to_tap_within_6h"]
    return [{"scenario": scenario, "seed": seed, "pair": c,
             "pct_agents": 100.0 * float(a[c].mean())} for c in cols]


def _agent_risk_rows(result, scenario, seed):
    """Risk cohorts built only from visits/movement observed before exposure.

    Day-3 and day-7 cohorts are constructed separately.  For an exposed agent,
    the window closes at the first S→E tick (including a causal tap visit at the
    same tick); for an unexposed agent it closes at the outcome horizon.  This
    removes post-exposure temporal leakage without changing the ABM.
    """
    init = result.get("initial_agents", pd.DataFrame()).rename(columns={"agent_id":"agent"}).copy()
    if init.empty:
        return [], [], pd.DataFrame()
    visits_all = result.get("completed_visits", pd.DataFrame()).copy()
    trips_all = result.get("actual_trips", pd.DataFrame()).copy()
    exp = result.get("exposures", pd.DataFrame()).copy()
    if exp is not None and not exp.empty:
        first_exp = exp.sort_values("tick").drop_duplicates("agent", keep="first")[["agent","tick","day","source"]]
        first_exp = first_exp.rename(columns={"tick":"exposure_tick","day":"exposure_day","source":"exposure_source"})
    else:
        first_exp = pd.DataFrame(columns=["agent","exposure_tick","exposure_day","exposure_source"])

    base = init.merge(first_exp, on="agent", how="left")
    base["became_exposed"] = base["exposure_tick"].notna()
    base["exposed_by_day1"] = base["exposure_day"].le(1)
    base["exposed_by_day2"] = base["exposure_day"].le(2)
    base["exposed_by_day3"] = base["exposure_day"].le(3)
    base["exposed_by_day7"] = base["exposure_day"].le(7)
    base["time_to_exposure_ticks"] = pd.to_numeric(base["exposure_tick"], errors="coerce")
    base["time_to_exposure_days"] = base["time_to_exposure_ticks"] / float(TICKS_PER_DAY)
    exp_tick_map = base.set_index("agent")["exposure_tick"]

    def _window_visits(horizon_day):
        horizon_tick = min(int(RUN_TICKS) - 1, int(horizon_day * TICKS_PER_DAY) - 1)
        if visits_all is None or visits_all.empty:
            return pd.DataFrame(columns=getattr(visits_all, 'columns', []))
        d = visits_all.copy()
        d["_exp_tick"] = d["agent"].map(exp_tick_map)
        keep = (pd.to_numeric(d["tick"], errors="coerce") <= horizon_tick) & (
            d["_exp_tick"].isna() | (pd.to_numeric(d["tick"], errors="coerce") <= d["_exp_tick"])
        )
        return d.loc[keep].drop(columns=["_exp_tick"])

    def _window_trips(horizon_day):
        horizon_tick = min(int(RUN_TICKS) - 1, int(horizon_day * TICKS_PER_DAY) - 1)
        if trips_all is None or trips_all.empty:
            return pd.DataFrame(columns=getattr(trips_all, 'columns', []))
        d = trips_all.copy()
        d["_exp_tick"] = d["agent"].map(exp_tick_map)
        keep = (pd.to_numeric(d["tick"], errors="coerce") <= horizon_tick) & (
            d["_exp_tick"].isna() | (pd.to_numeric(d["tick"], errors="coerce") <= d["_exp_tick"])
        )
        return d.loc[keep].drop(columns=["_exp_tick"])

    def _assign_cohort(a):
        act = a[["visited_school","visited_market","visited_generic_activity"]].any(axis=1)
        wash = a[["visited_tap","visited_toilet"]].any(axis=1)
        cohort = pd.Series("No activity or WASH visits", index=a.index, dtype=object)
        cohort.loc[act & ~wash] = "Activity only"
        cohort.loc[wash & ~act] = "WASH only"
        cohort.loc[act & wash] = "Activity + WASH, different days"
        cohort.loc[a["activity_wash_same_day"]] = "Activity + WASH, same day"
        cohort.loc[a["activity_to_tap_within_6h"]] = "Activity → tap within 6 h"
        return cohort

    horizon_tables = {}
    for horizon in (1, 2, 3, 7):
        vwin = _window_visits(horizon)
        awin = _agent_covisitation_table(result, visits_override=vwin)
        awin = awin.merge(base[["agent","household_id","home_x","home_y","exposure_tick","exposure_day","exposure_source","became_exposed","exposed_by_day1","exposed_by_day2","exposed_by_day3","exposed_by_day7","time_to_exposure_ticks","time_to_exposure_days"]], on="agent", how="left")
        twin = _window_trips(horizon)
        if twin is not None and not twin.empty:
            tg = twin.groupby("agent").agg(
                total_trip_steps=("tick","count"),
                total_distance_m=("actual_trip_distance_m","sum"),
            ).reset_index()
            awin = awin.merge(tg, on="agent", how="left")
        if "total_trip_steps" not in awin.columns:
            awin["total_trip_steps"] = 0.0
        if "total_distance_m" not in awin.columns:
            awin["total_distance_m"] = 0.0
        awin["total_trip_steps"] = pd.to_numeric(awin["total_trip_steps"], errors="coerce").fillna(0)
        awin["total_distance_m"] = pd.to_numeric(awin["total_distance_m"], errors="coerce").fillna(0)
        awin["risk_cohort"] = _assign_cohort(awin)
        horizon_tables[horizon] = awin

    rows = []
    for horizon, awin in horizon_tables.items():
        susceptible = awin[awin["initial_state"].eq("S")].copy()
        outcome_col = f"exposed_by_day{horizon}"
        for cohort, d in susceptible.groupby("risk_cohort"):
            row = {
                "scenario":scenario, "seed":seed, "risk_cohort":cohort,
                "horizon_day":horizon, "n_agents":len(d),
                "risk_day1_pct":np.nan, "risk_day2_pct":np.nan, "risk_day3_pct":np.nan, "risk_day7_pct":np.nan,
                "mean_exposure_tick":float(d.loc[d[outcome_col],"exposure_tick"].mean()) if d[outcome_col].any() else np.nan,
                "median_time_to_exposure_days":float(d.loc[d[outcome_col],"time_to_exposure_days"].median()) if d[outcome_col].any() else np.nan,
                "mean_trip_steps":float(d["total_trip_steps"].mean()),
                "mean_distance_km":float(d["total_distance_m"].mean()/1000.0),
            }
            row[f"risk_day{horizon}_pct"] = 100.0 * float(d[outcome_col].mean())
            rows.append(row)

    day7 = horizon_tables[7].copy()
    day7["scenario"] = scenario; day7["seed"] = seed
    day7["risk_cohort_day7"] = day7["risk_cohort"]
    day3_cohort = horizon_tables[3].set_index("agent")["risk_cohort"]
    day7["risk_cohort_day3"] = day7["agent"].map(day3_cohort)
    susceptible7 = day7[day7["initial_state"].eq("S")].copy()

    role_rows = []
    for (role, cohort), d in susceptible7.groupby(["role","risk_cohort_day7"]):
        role_rows.append({
            "scenario":scenario,"seed":seed,"role":role,"risk_cohort":cohort,"analysis_type":"broad_cohort",
            "n_agents":len(d),"risk_day2_pct":100.0*float(d["exposed_by_day2"].mean()),
            "risk_day3_pct":100.0*float(d["exposed_by_day3"].mean()),
            "risk_day7_pct":100.0*float(d["exposed_by_day7"].mean()),
            "median_time_to_exposure_days":float(d.loc[d["became_exposed"],"time_to_exposure_days"].median()) if d["became_exposed"].any() else np.nan,
            "mean_trip_steps":float(d["total_trip_steps"].mean()),
            "mean_distance_km":float(d["total_distance_m"].mean()/1000.0),
        })

    # Named within-role comparisons requested for interpretation.
    _role_specs = [
        ("student", "School + tap on same day", susceptible7["school_tap_same_day"]),
        ("student", "School without same-day tap", susceptible7["visited_school"] & ~susceptible7["school_tap_same_day"]),
        ("trader", "Market + tap on same day", susceptible7["market_tap_same_day"]),
        ("trader", "Market without same-day tap", susceptible7["visited_market"] & ~susceptible7["market_tap_same_day"]),
        ("general", "Activity + tap on same day", susceptible7["activity_tap_same_day"]),
        ("general", "Activity without same-day tap", susceptible7[["visited_school","visited_market","visited_generic_activity"]].any(axis=1) & ~susceptible7["activity_tap_same_day"]),
    ]
    for _role, _label, _mask in _role_specs:
        _d = susceptible7[susceptible7["role"].eq(_role) & _mask].copy()
        if not _d.empty:
            role_rows.append({
                "scenario":scenario,"seed":seed,"role":_role,"risk_cohort":_label,"analysis_type":"role_specific",
                "n_agents":len(_d),"risk_day2_pct":100.0*float(_d["exposed_by_day2"].mean()),
                "risk_day3_pct":100.0*float(_d["exposed_by_day3"].mean()),
                "risk_day7_pct":100.0*float(_d["exposed_by_day7"].mean()),
                "median_time_to_exposure_days":float(_d.loc[_d["became_exposed"],"time_to_exposure_days"].median()) if _d["became_exposed"].any() else np.nan,
                "mean_trip_steps":float(_d["total_trip_steps"].mean()),
                "mean_distance_km":float(_d["total_distance_m"].mean()/1000.0),
            })

    agent_cols = [
        "scenario","seed","agent","role","initial_state","risk_cohort_day3","risk_cohort_day7",
        "total_trip_steps","total_distance_m","became_exposed","exposed_by_day1","exposed_by_day2","exposed_by_day3",
        "exposed_by_day7","exposure_tick","exposure_day","time_to_exposure_ticks","time_to_exposure_days","exposure_source",
        "visited_school","visited_market","visited_generic_activity","visited_tap","visited_toilet",
        "school_tap_same_day","market_tap_same_day","activity_tap_same_day",
        "activity_wash_same_day","activity_to_tap_within_6h"
    ]
    for c in agent_cols:
        if c not in susceptible7.columns:
            susceptible7[c] = np.nan
    return rows, role_rows, susceptible7[agent_cols].copy()


def _tap_level_rows(result, scenario, seed):
    taps = _active_layer(result, "tap")
    if taps.empty:
        return []
    school_g = _active_layer(result,"school"); market_g = _active_layer(result,"market")
    generic_g = _active_layer(result,"generic_activity")
    home_g = _unique_home_gdf(result.get("initial_agents",pd.DataFrame()))
    schools = _coords_from_gdf(school_g); markets = _coords_from_gdf(market_g)
    generic = _coords_from_gdf(generic_g); homes = _coords_from_gdf(home_g)
    stree = cKDTree(schools) if len(schools) else None
    mtree = cKDTree(markets) if len(markets) else None
    gtree = cKDTree(generic) if len(generic) else None
    htree = cKDTree(homes) if len(homes) else None
    generic_weights = pd.to_numeric(generic_g.get("weight", pd.Series(np.ones(len(generic_g)))), errors="coerce").fillna(0).to_numpy(dtype=float) if len(generic_g) else np.asarray([], dtype=float)
    generic_total = float(generic_weights.sum())
    visits = result.get("completed_visits", pd.DataFrame()); exps = result.get("exposures", pd.DataFrame())
    rows=[]
    for _, r in taps.iterrows():
        xy=np.array([r.geometry.x,r.geometry.y]); fid=r.get("feature_id")
        dv = visits[(visits["facility_type"].eq("tap")) & (visits["facility_id"].eq(fid))] if visits is not None and not visits.empty else pd.DataFrame()
        de = exps[exps.get("exposure_facility_id", pd.Series(index=exps.index,dtype=object)).eq(fid)] if exps is not None and not exps.empty else pd.DataFrame()
        def _nearest(tree): return float(tree.query(xy,k=1)[0]) if tree is not None else np.nan
        def _count(tree): return int(len(tree.query_ball_point(xy,r=500))) if tree is not None else 0
        gidx = gtree.query_ball_point(xy,r=500) if gtree is not None else []
        gweight = float(generic_weights[np.asarray(gidx,dtype=int)].sum()) if len(gidx) else 0.0
        nearest_values = [_nearest(t) for t in (stree,mtree,gtree) if t is not None]
        n_school, n_market = _count(stree), _count(mtree)
        rows.append({
            "scenario":scenario,"seed":seed,"tap_id":fid,"tap_x":xy[0],"tap_y":xy[1],
            "nearest_activity_m":float(np.nanmin(nearest_values)) if nearest_values else np.nan,
            "nearest_school_m":_nearest(stree),"nearest_market_m":_nearest(mtree),"nearest_generic_m":_nearest(gtree),
            # Named amenities are counts; generic attraction is retained as a weight/share.
            "schools_within_500m":n_school,"markets_within_500m":n_market,
            "activity_within_500m":n_school+n_market,
            "generic_points_within_500m":int(len(gidx)),
            "generic_weight_within_500m":gweight,
            "generic_supply_share_within_500m_pct":100.0*gweight/generic_total if generic_total>0 else 0.0,
            "homes_within_500m":_count(htree),
            "nearest_home_m":_nearest(htree),
            "visits":int(len(dv)),"unique_visitors":int(dv["agent"].nunique()) if not dv.empty else 0,
            "tap_water_exposures":int(len(de)),
            "tap_water_exposures_per_1000_visits":float(1000.0*len(de)/max(len(dv),1)),
        })
    return rows




def _tap_contamination_rows(result, scenario, seed):
    """Return tap-level contamination provenance and use/exposure burden.

    These records describe contamination and tap use. They do not alter
    facility choice, exposure probabilities or the underlying ABM mechanisms.
    """
    recs = result.get("tap_contamination_records", pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
    if recs is None or getattr(recs, "empty", True):
        return []

    visits = result.get("completed_visits", pd.DataFrame())
    exps = result.get("exposures", pd.DataFrame())
    rows = []
    for _, rec in recs.iterrows():
        fid = rec.get("tap_id", None)
        if visits is not None and not getattr(visits, "empty", True):
            dv = visits[(visits["facility_type"].eq("tap")) & (visits["facility_id"].eq(fid))].copy()
        else:
            dv = pd.DataFrame()
        if exps is not None and not getattr(exps, "empty", True):
            _efid = exps.get("exposure_facility_id", pd.Series(index=exps.index, dtype=object))
            _src = exps.get("source", pd.Series(index=exps.index, dtype=object))
            de = exps[_efid.eq(fid) & _src.eq("tap_water")].copy()
        else:
            de = pd.DataFrame()

        origin = str(rec.get("contamination_origin", "never"))
        ctick = pd.to_numeric(pd.Series([rec.get("contamination_tick", np.nan)]), errors="coerce").iloc[0]
        if origin == "initial":
            dv_contam = dv
        elif origin == "dynamic" and np.isfinite(ctick) and not dv.empty:
            dv_ticks = pd.to_numeric(dv.get("tick", pd.Series(index=dv.index, dtype=float)), errors="coerce")
            # update_water_contamination is called after the tick's agent/exposure
            # updates. Visits strictly after ctick are therefore the visits that
            # occur while a dynamically contaminated tap is already flagged.
            dv_contam = dv.loc[dv_ticks.gt(float(ctick))].copy()
        else:
            dv_contam = dv.iloc[0:0].copy() if not dv.empty else pd.DataFrame()

        n_visits = int(len(dv))
        n_contam_visits = int(len(dv_contam))
        n_exp = int(len(de))
        rows.append({
            "scenario": scenario,
            "seed": int(seed),
            "tap_id": fid,
            "tap_x": float(rec.get("tap_x", np.nan)),
            "tap_y": float(rec.get("tap_y", np.nan)),
            "initially_contaminated": bool(rec.get("initially_contaminated", False)),
            "dynamically_contaminated": bool(rec.get("dynamically_contaminated", False)),
            "contamination_origin": origin,
            "contamination_tick": float(ctick) if np.isfinite(ctick) else np.nan,
            "contamination_day": ((float(ctick) + 1.0) / float(TICKS_PER_DAY)) if np.isfinite(ctick) else np.nan,
            "final_contaminated": bool(rec.get("final_contaminated", False)),
            "final_feces_load": float(rec.get("final_feces_load", np.nan)),
            "threshold_total": float(rec.get("threshold_total", np.nan)),
            "visits_total": n_visits,
            "unique_visitors_total": int(dv["agent"].nunique()) if not dv.empty and "agent" in dv.columns else 0,
            "visits_while_contaminated": n_contam_visits,
            "unique_visitors_while_contaminated": int(dv_contam["agent"].nunique()) if not dv_contam.empty and "agent" in dv_contam.columns else 0,
            "tap_water_exposures": n_exp,
            "tap_water_exposures_per_1000_contaminated_visits": (1000.0 * n_exp / n_contam_visits) if n_contam_visits > 0 else (0.0 if n_exp == 0 else np.nan),
        })
    return rows


def _zone_layer_rows(result, scenario, seed):
    zones=result.get("fixed_analysis_zones",None)
    if zones is None or getattr(zones,"empty",True): return [], []
    zones=zones.reset_index(drop=True).copy(); n=len(zones)
    data={"homes":np.zeros(n),"activity":np.zeros(n),"wash":np.zeros(n),"inbound_visits":np.zeros(n),"contaminated_tap_visits":np.zeros(n),"exposures":np.zeros(n)}
    homes=_unique_home_gdf(result.get("initial_agents",pd.DataFrame())); hc=_coords_from_gdf(homes)
    if len(hc):
        for i in _assign_points_to_polys(hc,zones): data["homes"][i]+=1
    for ft in ["generic_activity","school","market"]:
        g=_active_layer(result,ft); c=_coords_from_gdf(g)
        w=pd.to_numeric(g.get("weight",pd.Series(np.ones(len(g)))),errors="coerce").fillna(0).to_numpy() if len(g) else []
        for i,v in zip(_assign_points_to_polys(c,zones),w): data["activity"][i]+=v
    for ft in ["tap","toilet"]:
        c=_coords_from_gdf(_active_layer(result,ft))
        for i in _assign_points_to_polys(c,zones): data["wash"][i]+=1
    vis=result.get("completed_visits",pd.DataFrame())
    if vis is not None and not vis.empty:
        c=vis[["facility_x","facility_y"]].to_numpy(dtype=float)
        for i in _assign_points_to_polys(c,zones): data["inbound_visits"][i]+=1
        contam=set(result.get("contaminated_tap_wkbs",[]) or [])
        cv=vis[vis["facility_type"].eq("tap") & vis["facility_wkb"].isin(contam)]
        if not cv.empty:
            c=cv[["facility_x","facility_y"]].to_numpy(dtype=float)
            for i in _assign_points_to_polys(c,zones): data["contaminated_tap_visits"][i]+=1
    exp=result.get("exposures",pd.DataFrame())
    if exp is not None and not exp.empty:
        c=exp[["exposure_x","exposure_y"]].dropna().to_numpy(dtype=float)
        for i in _assign_points_to_polys(c,zones): data["exposures"][i]+=1
    rows=[]
    shares={}
    for layer,vals in data.items():
        sh=100*vals/vals.sum() if vals.sum()>0 else np.zeros_like(vals); shares[layer]=sh
        for zid,v in enumerate(sh): rows.append({"scenario":scenario,"seed":seed,"zone_id":zid,"layer":layer,"share_pct":float(v)})
    corr_pairs=[("activity","inbound_visits"),("wash","contaminated_tap_visits"),("inbound_visits","exposures"),("activity","exposures"),("wash","exposures")]
    corr=[]
    for a,b in corr_pairs:
        try: r=float(spearmanr(shares[a],shares[b],nan_policy="omit").statistic)
        except Exception: r=np.nan
        corr.append({"scenario":scenario,"seed":seed,"pair":f"{a} vs {b}","spearman_r":r})
    return rows,corr


def _legacy_ensemble_rows(result, scenario, seed, zone_rows):
    """Mean/SD-ready summaries of additional diagnostics."""
    N=max(int(result.get("N",1)),1); days=max(float(RUN_TICKS)/float(TICKS_PER_DAY),1.0)
    at=result.get("actual_trips",pd.DataFrame())
    moved=at[pd.to_numeric(at.get("actual_trip_distance_m",pd.Series(dtype=float)),errors="coerce").fillna(0)>0].copy() if at is not None and not at.empty else pd.DataFrame()
    trip={"scenario":scenario,"seed":seed,
          "trips_per_agent_day":float(len(moved)/(N*days)),
          "average_trip_length_km":float(pd.to_numeric(moved.get("actual_trip_distance_m",pd.Series(dtype=float)),errors="coerce").mean()/1000.0) if not moved.empty else 0.0,
          "pct_steps_same_zone":100.0*float((at["origin_zone"]==at["destination_zone"]).mean()) if at is not None and not at.empty and {"origin_zone","destination_zone"}.issubset(at.columns) else np.nan}
    _bins=["0-250m","250m-1km","1-3km","3-5km",">5km"]
    _den=max(len(moved),1)
    for _b in _bins:
        trip[f"distance_bin_{_b}_pct"]=100.0*float((moved.get("actual_moved_distance_bin",pd.Series(dtype=object))==_b).sum())/_den
    try:
        fb={"scenario":scenario,"seed":seed,**_facility_visit_burden_from_result(result,ticks=RUN_TICKS,ticks_per_day=TICKS_PER_DAY)}
    except Exception:
        fb={"scenario":scenario,"seed":seed}
    exp=result.get("exposures",pd.DataFrame()); src=[]
    total=max(len(exp),1) if exp is not None else 1
    for source in ["tap_water","home_water","household_secondary","unknown"]:
        n=int((exp["source"]==source).sum()) if exp is not None and not exp.empty and "source" in exp else 0
        src.append({"scenario":scenario,"seed":seed,"source":source,"count":n,"share_pct":100.0*n/total})
    zdf=pd.DataFrame(zone_rows)
    grav={"scenario":scenario,"seed":seed,"supply_inbound_spearman":np.nan,"dominant_zone_inbound_share_pct":np.nan}
    if not zdf.empty:
        p=zdf.pivot(index="zone_id",columns="layer",values="share_pct")
        if {"activity","inbound_visits"}.issubset(p.columns):
            try: grav["supply_inbound_spearman"]=float(spearmanr(p["activity"],p["inbound_visits"]).statistic)
            except Exception: pass
            grav["dominant_zone_inbound_share_pct"]=float(p["inbound_visits"].max())
    return trip,fb,src,grav


def _fixed_zone_od_matrix(result):
    zones=result.get("fixed_analysis_zones",None)
    n=0 if zones is None or getattr(zones,"empty",True) else len(zones)
    M=np.zeros((n,n),dtype=float)
    at=result.get("actual_trips",pd.DataFrame())
    if n==0 or at is None or at.empty or not {"origin_x","origin_y","destination_x","destination_y"}.issubset(at.columns): return M
    d=at[pd.to_numeric(at["actual_trip_distance_m"],errors="coerce").fillna(0)>0].dropna(subset=["origin_x","origin_y","destination_x","destination_y"])
    if d.empty: return M
    oi=_assign_points_to_polys(d[["origin_x","origin_y"]].to_numpy(dtype=float),zones)
    di=_assign_points_to_polys(d[["destination_x","destination_y"]].to_numpy(dtype=float),zones)
    for o,dd in zip(oi,di):
        if 0<=o<n and 0<=dd<n: M[o,dd]+=1
    if M.sum()>0: M=100.0*M/M.sum()
    return M


def _core_outcome_and_timeseries(result, scenario, seed):
    out=_extract_core_outcomes(result); out.update({"scenario":scenario,"seed":seed})
    trans=result.get("transitions",pd.DataFrame()); hist=result.get("state_history",pd.DataFrame()); N=max(int(result.get("N",1)),1)
    ts=[]
    if trans is not None and not trans.empty:
        se = pd.to_numeric(trans.get("SE", 0), errors="coerce").fillna(0)
        cum=100.0*se.cumsum()/N
        inc=100.0*se/N
        for t,v in enumerate(cum): ts.append({"scenario":scenario,"seed":seed,"tick":t,"metric":"cumulative_exposure_pct","value":float(v)})
        for t,v in enumerate(inc): ts.append({"scenario":scenario,"seed":seed,"tick":t,"metric":"incidence_pct","value":float(v)})
    if hist is not None and not hist.empty:
        I = pd.to_numeric(hist.get("I", 0), errors="coerce").fillna(0)
        E = pd.to_numeric(hist.get("E", 0), errors="coerce").fillna(0)
        for t,v in enumerate(100.0*I/N): ts.append({"scenario":scenario,"seed":seed,"tick":t,"metric":"active_infectious_pct","value":float(v)})
        for t,v in enumerate(100.0*(E+I)/N): ts.append({"scenario":scenario,"seed":seed,"tick":t,"metric":"active_exposed_infectious_pct","value":float(v)})
    foi=result.get("foi_tick",pd.DataFrame())
    if foi is not None and not foi.empty and "foi" in foi.columns:
        ticks_foi = pd.to_numeric(foi.get("tick", pd.Series(np.arange(len(foi)))), errors="coerce").fillna(0).astype(int)
        vals = pd.to_numeric(foi["foi"], errors="coerce")
        for t,v in zip(ticks_foi,vals):
            if np.isfinite(v): ts.append({"scenario":scenario,"seed":seed,"tick":int(t),"metric":"foi","value":float(v)})
    return out,ts


def _spatial_grid_setup(n=110):
    if CITY_POLY is not None:
        minx,miny,maxx,maxy=CITY_POLY.bounds
    else:
        minx,miny,maxx,maxy=city_bounds
    xe=np.linspace(minx,maxx,n+1); ye=np.linspace(miny,maxy,n+1)
    return xe,ye


def _density_field(coords, xe, ye, sigma_cells=1.6):
    if coords is None or len(coords)==0: return np.zeros((len(ye)-1,len(xe)-1),dtype=float)
    H,_,_=np.histogram2d(coords[:,1],coords[:,0],bins=[ye,xe])
    return gaussian_filter(H.astype(float),sigma=sigma_cells,mode="constant")


def _annotated_heatmap(pivot_mean,pivot_sd,title,outfile,cmap="viridis",fmt=".2f",cbar_label=None):
    rows=list(pivot_mean.index); cols=list(pivot_mean.columns)
    arr=pivot_mean.to_numpy(dtype=float); sd=pivot_sd.reindex(index=rows,columns=cols).to_numpy(dtype=float)
    fig,ax=plt.subplots(figsize=(max(8,1.6*len(cols)),max(4,0.65*len(rows))))
    im=ax.imshow(arr,aspect="auto",cmap=cmap)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=35,ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i,j]): ax.text(j,i,f"{arr[i,j]:{fmt}}\n±{sd[i,j]:{fmt}}",ha="center",va="center",fontsize=8)
    cb=fig.colorbar(im,ax=ax); cb.set_label(cbar_label or "Mean across seeds")
    ax.set_title(title); fig.tight_layout(); fig.savefig(outfile,dpi=250,bbox_inches="tight"); plt.close(fig)


def _save_ensemble_plots(tables, spatial_acc, layout_results):
    print("[Plots 1/2] Saving core ensemble and retained legacy-equivalent figures...")
    # Disease outcomes bars
    d=tables["outcomes"]
    metrics=[("cumulative_exposure_attack_rate_pct","Final cumulative exposure attack rate (%)"),
             ("peak_6hr_incidence_pct","Peak 6-hour incidence (%)"),
             ("peak_active_infectious_pct","Peak active infectious prevalence (%)")]
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    for ax,(m,title) in zip(axs,metrics):
        g=d.groupby("scenario")[m].agg(["mean","std"]).reindex(COOCC_ORDER)
        x=np.arange(len(g)); ax.bar(x,g["mean"],yerr=g["std"],capsize=4)
        ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[k] for k in g.index],rotation=25,ha="right")
        ax.set_title(title); ax.set_ylabel("Mean ± SD across seeds")
        for i,v in enumerate(g["mean"]): ax.text(i,v,f"{v:.1f}\n±{g['std'].iloc[i]:.1f}",ha="center",va="bottom",fontsize=8)
    fig.suptitle(f"Decentralisation outcomes across {COOCCURRENCE_N_SEEDS} seeds"); fig.tight_layout()
    fig.savefig(COOCC_OUT/f"{CITY_SLUG}_ensemble_intervention_summary.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    # Time series shaded
    ts=tables["timeseries"]
    for metric,title,ylabel in [("cumulative_exposure_pct","Cumulative exposure attack rate","% of agents"),("active_infectious_pct","Active infectious prevalence","% of agents")]:
        fig,ax=plt.subplots(figsize=(10,6))
        z=ts[ts.metric.eq(metric)]
        for sc in COOCC_ORDER:
            g=z[z.scenario.eq(sc)].groupby("tick")["value"].agg(["mean","std"])
            ax.plot(g.index,g["mean"],label=COOCC_LABELS[sc]); ax.fill_between(g.index,g["mean"]-g["std"],g["mean"]+g["std"],alpha=.18)
        ax.set_title(f"{title}: mean ± SD across {COOCCURRENCE_N_SEEDS} seeds"); ax.set_xlabel("Tick"); ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout()
        fig.savefig(COOCC_OUT/f"{CITY_SLUG}_{metric}_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    # Physical co-occurrence at 500 m
    pc=tables["physical"].copy(); pc=pc[np.isclose(pc.radius_m,500)].copy()
    pc["pair"]=pc.feature_A+" – "+pc.feature_B
    gm=pc.groupby(["pair","scenario"])["pct_A_with_B_in_radius"].mean().unstack().reindex(columns=COOCC_ORDER)
    gs=pc.groupby(["pair","scenario"])["pct_A_with_B_in_radius"].std().unstack().reindex(index=gm.index,columns=gm.columns).fillna(0)
    gm.columns=[COOCC_LABELS[c] for c in gm.columns]; gs.columns=gm.columns
    _annotated_heatmap(gm,gs,"Physical feature co-occurrence: % of A with B within 500 m",COOCC_OUT/f"{CITY_SLUG}_physical_cooccurrence_500m_mean_sd.png",cbar_label="Percent")

    # Urban form
    uf=tables["urban_form"]
    uf["metric_label"]=uf.layer+" | entropy"
    gm=uf.groupby(["scenario","metric_label"])["normalised_entropy"].mean().unstack().reindex(COOCC_ORDER)
    gs=uf.groupby(["scenario","metric_label"])["normalised_entropy"].std().unstack().reindex(index=gm.index,columns=gm.columns).fillna(0)
    gm.index=[COOCC_LABELS[i] for i in gm.index]; gs.index=gm.index
    _annotated_heatmap(gm,gs,"Monocentric–polycentric structure by layer (normalised entropy)",COOCC_OUT/f"{CITY_SLUG}_urban_form_entropy_mean_sd.png",cbar_label="Entropy (0 concentrated, 1 even)")

    # Attraction heatmaps
    at=tables["attraction"]
    for metric,title,suffix in [("visits_per_agent_day","Amenity visits per agent per day","amenity_visits"),("median_trip_distance_m","Median distance travelled to each amenity (m)","amenity_distance")]:
        gm=at.groupby(["scenario","facility_type"])[metric].mean().unstack().reindex(index=COOCC_ORDER,columns=COOCC_FEATURE_ORDER)
        gs=at.groupby(["scenario","facility_type"])[metric].std().unstack().reindex(index=gm.index,columns=gm.columns).fillna(0)
        gm.index=[COOCC_LABELS[i] for i in gm.index]; gs.index=gm.index
        _annotated_heatmap(gm,gs,title,COOCC_OUT/f"{CITY_SLUG}_{suffix}_mean_sd.png",cbar_label="Mean across seeds")

    # Co-presence/contact opportunities (not direct transmission in the current model)
    ct=tables["contact"]
    if not ct.empty:
        gm=ct.groupby(["scenario","facility_type"])["pct_visits_with_copresence"].mean().unstack().reindex(index=COOCC_ORDER,columns=COOCC_FEATURE_ORDER)
        gs=ct.groupby(["scenario","facility_type"])["pct_visits_with_copresence"].std().unstack().reindex(index=gm.index,columns=gm.columns).fillna(0)
        gm.index=[COOCC_LABELS[i] for i in gm.index]; gs.index=gm.index
        _annotated_heatmap(gm,gs,"Co-presence opportunity: visits occurring with another agent",COOCC_OUT/f"{CITY_SLUG}_facility_copresence_mean_sd.png",cbar_label="Percent of visits")

    # Co-visitation
    cv=tables["covisitation"]
    gm=cv.groupby(["pair","scenario"])["pct_agents"].mean().unstack().reindex(columns=COOCC_ORDER)
    gs=cv.groupby(["pair","scenario"])["pct_agents"].std().unstack().reindex(index=gm.index,columns=gm.columns).fillna(0)
    gm.columns=[COOCC_LABELS[c] for c in gm.columns]; gs.columns=gm.columns
    _annotated_heatmap(gm,gs,"Mobility-mediated co-occurrence: agents experiencing each combination",COOCC_OUT/f"{CITY_SLUG}_covisitation_mean_sd.png",cbar_label="Percent of agents")

    # Zone correlation
    zc=tables["zone_corr"]
    gm=zc.groupby(["scenario","pair"])["spearman_r"].mean().unstack().reindex(COOCC_ORDER)
    gs=zc.groupby(["scenario","pair"])["spearman_r"].std().unstack().reindex(index=gm.index,columns=gm.columns).fillna(0)
    gm.index=[COOCC_LABELS[i] for i in gm.index]; gs.index=gm.index
    _annotated_heatmap(gm,gs,"Zone-level co-occurrence correlations",COOCC_OUT/f"{CITY_SLUG}_zone_layer_correlations_mean_sd.png",cmap="coolwarm",cbar_label="Spearman r")

    # Risk cohorts
    risk=tables["risk"]
    if not risk.empty:
        cohorts=list(risk.risk_cohort.drop_duplicates())
        fig,axs=plt.subplots(1,2,figsize=(17,6))
        for ax,metric,title in [(axs[0],"risk_day3_pct","Exposure by day 3"),(axs[1],"risk_day7_pct","Exposure by day 7")]:
            width=.18; x=np.arange(len(cohorts))
            for j,sc in enumerate(COOCC_ORDER):
                g=risk[risk.scenario.eq(sc)].groupby("risk_cohort")[metric].agg(["mean","std"]).reindex(cohorts)
                ax.bar(x+(j-1.5)*width,g["mean"],width,yerr=g["std"],capsize=2,label=COOCC_LABELS[sc])
            ax.set_xticks(x); ax.set_xticklabels(cohorts,rotation=30,ha="right"); ax.set_ylabel("Risk (%) mean ± SD"); ax.set_title(title)
        axs[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_risk_by_covisitation_cohort_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    # Travel intensity versus individual exposure risk (initially susceptible only)
    ar=tables["agent_risk"]
    if not ar.empty:
        ar=ar.copy()
        ar["distance_quartile"] = ar.groupby(["scenario","seed"])["total_distance_m"].transform(
            lambda x: pd.qcut(x.rank(method="first"),4,labels=["Q1 lowest","Q2","Q3","Q4 highest"])
        )
        tq=ar.groupby(["scenario","seed","distance_quartile"],observed=True)["exposed_by_day7"].mean().mul(100).reset_index(name="risk_pct")
        fig,ax=plt.subplots(figsize=(10,5)); qs=["Q1 lowest","Q2","Q3","Q4 highest"]; x=np.arange(4); width=.18
        for j,sc in enumerate(COOCC_ORDER):
            g=tq[tq.scenario.eq(sc)].groupby("distance_quartile",observed=True)["risk_pct"].agg(["mean","std"]).reindex(qs)
            ax.bar(x+(j-1.5)*width,g["mean"],width,yerr=g["std"],capsize=2,label=COOCC_LABELS[sc])
        ax.set_xticks(x); ax.set_xticklabels(qs); ax.set_ylabel("Exposure by day 7 (%) mean ± SD"); ax.set_title("Travel distance and individual exposure risk"); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(COOCC_OUT/f"{CITY_SLUG}_travel_distance_risk_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    # Tap proximity versus burden/exposure
    tap=tables["tap_level"]
    if not tap.empty:
        agg=tap.groupby(["scenario","tap_id","activity_within_500m","homes_within_500m","nearest_activity_m"]).agg(mean_exposures=("tap_water_exposures","mean"),sd_exposures=("tap_water_exposures","std"),mean_visits=("visits","mean"),sd_visits=("visits","std")).reset_index().fillna(0)
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        for sc in COOCC_ORDER:
            dsc=agg[agg.scenario.eq(sc)]
            axs[0].errorbar(dsc.activity_within_500m,dsc.mean_visits,yerr=dsc.sd_visits,fmt='o',label=COOCC_LABELS[sc],alpha=.75,capsize=2)
            axs[1].errorbar(dsc.activity_within_500m,dsc.mean_exposures,yerr=dsc.sd_exposures,fmt='o',label=COOCC_LABELS[sc],alpha=.75,capsize=2)
        axs[0].set(xlabel="Activity amenities within 500 m",ylabel="Mean tap visits",title="Activity proximity and tap burden")
        axs[1].set(xlabel="Activity amenities within 500 m",ylabel="Mean tap-water exposures",title="Activity proximity and tap exposure")
        axs[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_tap_proximity_burden_exposure.png",dpi=250,bbox_inches="tight"); plt.close(fig)

        # Residential versus activity-linked placement profile.
        profile=tap.groupby(["scenario","seed"]).agg(residential_alignment=("homes_within_500m","mean"),activity_alignment=("activity_within_500m","mean")).reset_index()
        fig,axs=plt.subplots(1,2,figsize=(11,4.5))
        for ax,metric,title in [(axs[0],"residential_alignment","Mean homes within 500 m of a tap"),(axs[1],"activity_alignment","Mean activity amenities within 500 m of a tap")]:
            g=profile.groupby("scenario")[metric].agg(["mean","std"]).reindex(COOCC_ORDER).fillna(0)
            x=np.arange(len(g)); ax.bar(x,g["mean"],yerr=g["std"],capsize=3); ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[i] for i in g.index],rotation=25,ha="right"); ax.set_title(title); ax.set_ylabel("Mean ± SD")
        fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_water_placement_alignment_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    # Ensemble summaries of trips, facilities, exposure sources and gravity metrics.
    lt=tables["legacy_trip"]
    if not lt.empty:
        metrics=[("trips_per_agent_day","Trips / agent / day"),("average_trip_length_km","Average trip length (km)"),("pct_steps_same_zone","% movement steps within same zone")]
        fig,axs=plt.subplots(1,3,figsize=(16,4.8))
        for ax,(m,title) in zip(axs,metrics):
            g=lt.groupby("scenario")[m].agg(["mean","std"]).reindex(COOCC_ORDER).fillna(0); x=np.arange(len(g)); ax.bar(x,g["mean"],yerr=g["std"],capsize=3); ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[i] for i in g.index],rotation=25,ha="right"); ax.set_title(title); ax.set_ylabel("Mean ± SD")
        fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_legacy_trip_diagnostics_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)
        _bcols=[c for c in lt.columns if c.startswith("distance_bin_") and c.endswith("_pct")]
        if _bcols:
            fig,ax=plt.subplots(figsize=(12,5)); x=np.arange(len(_bcols)); width=.18
            for j,sc in enumerate(COOCC_ORDER):
                d=lt[lt.scenario.eq(sc)]; means=[d[c].mean() for c in _bcols]; sds=[d[c].std() for c in _bcols]
                ax.bar(x+(j-1.5)*width,means,width,yerr=sds,capsize=2,label=COOCC_LABELS[sc])
            ax.set_xticks(x); ax.set_xticklabels([c.replace("distance_bin_","").replace("_pct","") for c in _bcols]); ax.set_ylabel("Movement-step share (%) mean ± SD"); ax.set_title("Movement-distance distribution"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_legacy_trip_distance_bins_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    lf=tables["legacy_facility"]
    if not lf.empty:
        wanted=[c for c in ["tap_visits_per_agent_per_day","tap_share_top3_pct","tap_visit_gini","tap_unique_visited","tap_avg_agents_per_tap","tap_avg_visits_per_contaminated_tap","toilet_visits_per_agent_per_day"] if c in lf.columns]
        if wanted:
            fig,axs=plt.subplots(2,4,figsize=(17,9)); axs=np.ravel(axs)
            for ax,m in zip(axs,wanted):
                g=lf.groupby("scenario")[m].agg(["mean","std"]).reindex(COOCC_ORDER).fillna(0); x=np.arange(len(g)); ax.bar(x,g["mean"],yerr=g["std"],capsize=3); ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[i] for i in g.index],rotation=25,ha="right",fontsize=8); ax.set_title(m.replace("_"," "))
            for ax in axs[len(wanted):]: ax.axis("off")
            fig.suptitle("Local facility burden: mean ± SD across seeds"); fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_legacy_facility_burden_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    es=tables["exposure_source"]
    if not es.empty:
        fig,ax=plt.subplots(figsize=(10,5)); sources=["tap_water","home_water","household_secondary","unknown"]; x=np.arange(len(COOCC_ORDER)); width=.18
        for j,src in enumerate(sources):
            g=es[es.source.eq(src)].groupby("scenario")["share_pct"].agg(["mean","std"]).reindex(COOCC_ORDER).fillna(0); ax.bar(x+(j-1.5)*width,g["mean"],width,yerr=g["std"],capsize=2,label=src)
        ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[i] for i in COOCC_ORDER],rotation=20,ha="right"); ax.set_ylabel("Exposure source share (%) mean ± SD"); ax.legend(); fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_legacy_exposure_sources_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    gr=tables["gravity"]
    if not gr.empty:
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        for ax,m,title in [(axs[0],"supply_inbound_spearman","Activity supply → realised visits correlation"),(axs[1],"dominant_zone_inbound_share_pct","Dominant-zone inbound visit share (%)")]:
            g=gr.groupby("scenario")[m].agg(["mean","std"]).reindex(COOCC_ORDER).fillna(0); x=np.arange(len(g)); ax.bar(x,g["mean"],yerr=g["std"],capsize=3); ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[i] for i in g.index],rotation=25,ha="right"); ax.set_title(title); ax.set_ylabel("Mean ± SD")
        fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_legacy_gravity_response_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    # Mean and SD OD matrices on fixed baseline zones.
    fig,axs=plt.subplots(2,4,figsize=(18,9))
    vmax=0; sdmax=0; odmean={}; odsd={}
    for sc in COOCC_ORDER:
        stack=np.stack(spatial_acc["od"][sc],axis=0); odmean[sc]=stack.mean(axis=0); odsd[sc]=stack.std(axis=0,ddof=1) if len(stack)>1 else np.zeros_like(stack[0]); vmax=max(vmax,float(odmean[sc].max())); sdmax=max(sdmax,float(odsd[sc].max()))
    for j,sc in enumerate(COOCC_ORDER):
        im=axs[0,j].imshow(odmean[sc],cmap="Blues",vmin=0,vmax=max(vmax,1e-9)); axs[0,j].set_title(COOCC_LABELS[sc]); axs[0,j].set_xlabel("Destination zone"); axs[0,j].set_ylabel("Origin zone")
        im2=axs[1,j].imshow(odsd[sc],cmap="Oranges",vmin=0,vmax=max(sdmax,1e-9)); axs[1,j].set_xlabel("Destination zone"); axs[1,j].set_ylabel("Origin zone")
    fig.colorbar(im,ax=axs[0,:].tolist(),fraction=.02,pad=.01,label="Mean % of movements"); fig.colorbar(im2,ax=axs[1,:].tolist(),fraction=.02,pad=.01,label="SD across seeds"); fig.suptitle("Fixed-zone OD matrices: mean and SD"); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_legacy_OD_heatmaps_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    # Static spatial systems map. No SD is shown because urban layouts are held fixed across ABM seeds.
    fig,axs=plt.subplots(1,4,figsize=(20,5.5))
    for ax,sc in zip(axs,COOCC_ORDER):
        res=layout_results[sc]; outer=res.get("outer_cells_active")
        _draw_city_outline(ax, {"fixed_analysis_zones": outer}, color="0.35", linewidth=.7, alpha=.85, zorder=2)
        homes=_unique_home_gdf(res.get("initial_agents",pd.DataFrame()))
        if not homes.empty: homes.plot(ax=ax,color="0.75",markersize=2,alpha=.35)
        _gg=_active_layer(res,"generic_activity")
        if not _gg.empty:
            _gg.plot(ax=ax,color="orange",markersize=3,alpha=.18,label="generic activity")
        styles={"school":("^","purple"),"market":("s","green"),"tap":("o","deepskyblue"),"toilet":("x","black")}
        for ft,(marker,color) in styles.items():
            g=_active_layer(res,ft)
            if not g.empty: g.plot(ax=ax,marker=marker,color=color,markersize=25,label=ft)
        ax.set_title(COOCC_LABELS[sc]); ax.axis("off")
    axs[-1].legend(fontsize=7,loc="lower right"); fig.suptitle("Scenario spatial systems (fixed layout across stochastic seeds)"); fig.tight_layout()
    fig.savefig(COOCC_OUT/f"{CITY_SLUG}_scenario_spatial_systems.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # Spatial mean and SD fields for actual exposure events and peak E+I locations.
    xe,ye=spatial_acc["xe"],spatial_acc["ye"]; extent=[xe[0],xe[-1],ye[0],ye[-1]]
    for field_key,title,suffix in [("exposure","S→E exposure-event density","exposure_density"),("initial_ei","Initial E+I density","initial_EI_density"),("peak_ei","Peak E+I density","peak_EI_density"),("final_ei","Final E+I density","final_EI_density")]:
        means={}; sds={}; vmax=0; sdmax=0
        for sc in COOCC_ORDER:
            stack=np.stack(spatial_acc[field_key][sc],axis=0); means[sc]=stack.mean(axis=0); sds[sc]=stack.std(axis=0,ddof=1) if len(stack)>1 else np.zeros_like(stack[0]); vmax=max(vmax,float(means[sc].max())); sdmax=max(sdmax,float(sds[sc].max()))
        fig,axs=plt.subplots(2,4,figsize=(20,10),sharex=True,sharey=True)
        for j,sc in enumerate(COOCC_ORDER):
            im=axs[0,j].imshow(means[sc],origin="lower",extent=extent,cmap="turbo",vmin=0,vmax=max(vmax,1e-9)); axs[0,j].set_title(COOCC_LABELS[sc]);
            im2=axs[1,j].imshow(sds[sc],origin="lower",extent=extent,cmap="magma",vmin=0,vmax=max(sdmax,1e-9));
            _lr=layout_results[sc]; _h=_lr.get("inner_hubs_active"); _o=_lr.get("outer_cells_active")
            for _ax in (axs[0,j],axs[1,j]):
                _draw_city_outline(_ax, {"fixed_analysis_zones": _o}, color="white", linewidth=.7, alpha=.85, zorder=4)
                _gg=_active_layer(_lr,"generic_activity")
                if not _gg.empty: _gg.plot(ax=_ax,color="orange",markersize=2,alpha=.12)
                for _ft,_mk,_co in [("school","^","purple"),("market","s","lime"),("tap","o","cyan")]:
                    _g=_active_layer(_lr,_ft)
                    if not _g.empty: _g.plot(ax=_ax,marker=_mk,color=_co,markersize=8,alpha=.65)
                _ax.axis("off")
        axs[0,0].set_ylabel("Mean"); axs[1,0].set_ylabel("SD"); fig.colorbar(im,ax=axs[0,:].tolist(),fraction=.015,pad=.01,label="Mean density"); fig.colorbar(im2,ax=axs[1,:].tolist(),fraction=.015,pad=.01,label="SD across seeds")
        fig.suptitle(f"{title}: mean and SD across {COOCCURRENCE_N_SEEDS} seeds"); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_{suffix}_spatial_mean_sd.png",dpi=300,bbox_inches="tight"); plt.close(fig)



def _heatmap_values_only(pivot, title, outfile, cmap="viridis", fmt=".2f", cbar_label=None):
    """Heatmap for fixed structural layouts where seed SD is exactly zero by design."""
    if pivot is None or pivot.empty:
        return
    arr=pivot.to_numpy(dtype=float); rows=list(pivot.index); cols=list(pivot.columns)
    fig,ax=plt.subplots(figsize=(max(8,1.6*len(cols)),max(4,0.65*len(rows))))
    im=ax.imshow(arr,aspect="auto",cmap=cmap)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=35,ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    threshold=np.nanmedian(arr) if np.isfinite(arr).any() else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val=arr[i,j]
            ax.text(j,i,"NA" if not np.isfinite(val) else format(val,fmt),ha="center",va="center",fontsize=8,color="white" if np.isfinite(val) and val>threshold else "black")
    ax.set_title(title)
    cb=fig.colorbar(im,ax=ax); cb.set_label(cbar_label or "Value")
    fig.tight_layout(); fig.savefig(outfile,dpi=260,bbox_inches="tight"); plt.close(fig)


def _save_additional_ensemble_plots(tables, spatial_acc, layout_results):
    """Paper-ready plots for every new analysis table, in addition to old plots."""
    print("[Plots 2/2] Saving the full co-occurrence, risk, tap-placement and structural figure suite...")
    # 1) Physical co-occurrence at all radii, and nearest-distance matrix.
    pc=tables.get("physical",pd.DataFrame()).copy()
    if not pc.empty:
        pc["pair"]=pc.feature_A+" – "+pc.feature_B
        # Layout is fixed, so show values once rather than meaningless ±0 SD.
        fixed=pc.groupby(["pair","scenario","radius_m"],dropna=False).first().reset_index()
        fig,axs=plt.subplots(1,3,figsize=(18,6),sharey=True)
        for ax,r in zip(axs,[250.0,500.0,1000.0]):
            d=fixed[np.isclose(fixed.radius_m,r)]
            pv=d.pivot(index="pair",columns="scenario",values="pct_A_with_B_in_radius").reindex(columns=COOCC_ORDER)
            arr=pv.to_numpy(dtype=float); im=ax.imshow(arr,aspect="auto",cmap="viridis",vmin=0,vmax=100)
            ax.set_title(f"Within {int(r)} m"); ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels([COOCC_LABELS[c] for c in pv.columns],rotation=30,ha="right")
            ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index if ax is axs[0] else [])
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if np.isfinite(arr[i,j]): ax.text(j,i,f"{arr[i,j]:.1f}",ha="center",va="center",fontsize=7,color="white" if arr[i,j]>50 else "black")
        fig.colorbar(im,ax=axs.tolist(),fraction=.02,pad=.02,label="% of A with B in buffer")
        fig.suptitle("Physical/environmental co-occurrence across buffer sizes (fixed layouts)")
        fig.savefig(COOCC_OUT/f"{CITY_SLUG}_physical_cooccurrence_all_radii.png",dpi=260,bbox_inches="tight"); plt.close(fig)
        nd=fixed[np.isclose(fixed.radius_m,500)].pivot(index="pair",columns="scenario",values="median_nearest_distance_m").reindex(columns=COOCC_ORDER)
        nd.columns=[COOCC_LABELS[c] for c in nd.columns]
        _heatmap_values_only(nd,"Median nearest-neighbour distance between feature pairs",COOCC_OUT/f"{CITY_SLUG}_physical_nearest_distance.png",cbar_label="Distance (m)")

    # 2) Structural form: fixed supply layers without SD; realised-use layer with SD.
    uf=tables.get("urban_form",pd.DataFrame()).copy()
    if not uf.empty:
        for metric,title,suffix in [
            ("top1_share_pct","Largest-hub share by urban layer","top1"),
            ("normalised_entropy","Polycentricity/evenness by urban layer","entropy"),
            ("gini","Inequality by urban layer","gini"),
        ]:
            fixed=uf[uf.layer.ne("realised_visits")].groupby(["scenario","layer"])[metric].first().unstack().reindex(COOCC_ORDER)
            fixed.index=[COOCC_LABELS[i] for i in fixed.index]
            _heatmap_values_only(fixed,title+" (fixed layouts)",COOCC_OUT/f"{CITY_SLUG}_urban_form_{suffix}_fixed.png",cbar_label=metric.replace("_"," "))
        rv=uf[uf.layer.eq("realised_visits")]
        if not rv.empty:
            fig,axs=plt.subplots(1,3,figsize=(14,4.5))
            for ax,(m,title) in zip(axs,[("top1_share_pct","Largest hub visit share"),("normalised_entropy","Visit entropy"),("gini","Visit Gini")]):
                g=rv.groupby("scenario")[m].agg(["mean","std"]).reindex(COOCC_ORDER).fillna(0); x=np.arange(len(g))
                ax.bar(x,g["mean"],yerr=g["std"],capsize=3); ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[s] for s in g.index],rotation=25,ha="right"); ax.set_title(title); ax.set_ylabel("Mean ± SD")
            fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_functional_polycentricity_mean_sd.png",dpi=260,bbox_inches="tight"); plt.close(fig)

    # 3) Every amenity-attraction metric.
    at=tables.get("attraction",pd.DataFrame()).copy()
    if not at.empty:
        metrics=[
            ("total_visits","Total completed visits"),("unique_visitors","Unique visitors"),
            ("visits_per_facility","Visits per facility"),("mean_trip_distance_m","Mean completed-journey distance (m)"),
            ("median_trip_distance_m","Median completed-journey distance (m)"),("mean_travel_time_ticks","Mean travel time (ticks)"),
            ("top1_visit_share_pct","Top-facility visit share (%)"),("top3_visit_share_pct","Top-three facility visit share (%)"),
            ("visit_gini","Facility visit Gini"),
        ]
        fig,axs=plt.subplots(3,3,figsize=(18,15)); axs=np.ravel(axs)
        for ax,(m,title) in zip(axs,metrics):
            gm=at.groupby(["scenario","facility_type"])[m].mean().unstack().reindex(index=COOCC_ORDER,columns=COOCC_FEATURE_ORDER)
            arr=gm.to_numpy(dtype=float); im=ax.imshow(arr,aspect="auto",cmap="viridis")
            ax.set_title(title,fontsize=10); ax.set_xticks(range(len(gm.columns))); ax.set_xticklabels(gm.columns,rotation=35,ha="right",fontsize=8); ax.set_yticks(range(len(gm.index))); ax.set_yticklabels([COOCC_LABELS[s] for s in gm.index],fontsize=8)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if np.isfinite(arr[i,j]): ax.text(j,i,f"{arr[i,j]:.2f}",ha="center",va="center",fontsize=6)
            fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
        fig.suptitle("Amenity attraction, access and concentration: means across seeds"); fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_amenity_attraction_all_metrics.png",dpi=260,bbox_inches="tight"); plt.close(fig)

    # 4) All co-presence metrics.
    ct=tables.get("contact",pd.DataFrame()).copy()
    if not ct.empty:
        fig,axs=plt.subplots(1,3,figsize=(17,5))
        for ax,(m,title) in zip(axs,[("pct_visits_with_copresence","Visits with co-presence (%)"),("mean_agents_per_occupied_event","Agents per occupied facility-tick"),("contact_pairs_per_agent_day","Potential contact pairs / agent / day")]):
            g=ct.groupby(["scenario","facility_type"])[m].mean().unstack().reindex(index=COOCC_ORDER,columns=COOCC_FEATURE_ORDER); arr=g.to_numpy(dtype=float); im=ax.imshow(arr,aspect="auto",cmap="magma")
            ax.set_title(title,fontsize=10); ax.set_xticks(range(len(g.columns))); ax.set_xticklabels(g.columns,rotation=35,ha="right"); ax.set_yticks(range(len(g.index))); ax.set_yticklabels([COOCC_LABELS[s] for s in g.index]); fig.colorbar(im,ax=ax,fraction=.045,pad=.03)
        fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_facility_copresence_all_metrics.png",dpi=260,bbox_inches="tight"); plt.close(fig)

    # 5) Role-specific risk (students/traders/general) and movement intensity.
    rr=tables.get("role_risk",pd.DataFrame()).copy()
    if not rr.empty:
        if "analysis_type" in rr.columns and (rr["analysis_type"]=="role_specific").any():
            rr=rr[rr["analysis_type"]=="role_specific"].copy()
        roles=[r for r in ["student","trader","general"] if r in set(rr.role)]
        if not roles:
            roles=sorted(rr.role.dropna().unique())
        fig,axs=plt.subplots(len(roles),1,figsize=(16,max(5,4.5*len(roles))),squeeze=False)
        cohorts=list(rr.risk_cohort.drop_duplicates())
        for ax,role in zip(axs[:,0],roles):
            x=np.arange(len(cohorts)); width=.18
            for j,sc in enumerate(COOCC_ORDER):
                g=rr[(rr.role.eq(role))&(rr.scenario.eq(sc))].groupby("risk_cohort")["risk_day7_pct"].agg(["mean","std"]).reindex(cohorts)
                ax.bar(x+(j-1.5)*width,g["mean"],width,yerr=g["std"],capsize=2,label=COOCC_LABELS[sc])
            ax.set_xticks(x); ax.set_xticklabels(cohorts,rotation=25,ha="right"); ax.set_ylabel("Day-7 exposure risk (%)"); ax.set_title(f"{role.title()} agents: pre-exposure co-visitation and risk")
        axs[0,0].legend(fontsize=8,ncol=4); fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_role_specific_risk_mean_sd.png",dpi=260,bbox_inches="tight"); plt.close(fig)

    # 6) Tap placement components separately, avoiding generic-node count inflation.
    tap=tables.get("tap_level",pd.DataFrame()).copy()
    if not tap.empty:
        profile=tap.groupby(["scenario","seed"]).agg(
            homes=("homes_within_500m","mean"), schools=("schools_within_500m","mean"),
            markets=("markets_within_500m","mean"), generic_supply=("generic_supply_share_within_500m_pct","mean")
        ).reset_index()
        fig,axs=plt.subplots(1,4,figsize=(19,4.8))
        for ax,(m,title) in zip(axs,[("homes","Homes within 500 m"),("schools","Schools within 500 m"),("markets","Markets within 500 m"),("generic_supply","Generic supply share within 500 m (%)")]):
            g=profile.groupby("scenario")[m].agg(["mean","std"]).reindex(COOCC_ORDER).fillna(0); x=np.arange(len(g)); ax.bar(x,g["mean"],yerr=g["std"],capsize=3); ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[s] for s in g.index],rotation=28,ha="right"); ax.set_title(title); ax.set_ylabel("Mean ± SD")
        fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_tap_placement_components_mean_sd.png",dpi=260,bbox_inches="tight"); plt.close(fig)
        # Distance-band burden and exposure.
        tap["activity_distance_band"]=pd.cut(tap["nearest_activity_m"],bins=[-np.inf,250,500,1000,np.inf],labels=["≤250 m","250–500 m","500–1000 m",">1000 m"])
        band=tap.groupby(["scenario","seed","activity_distance_band"],observed=True).agg(visits=("visits","mean"),exposures=("tap_water_exposures","mean")).reset_index()
        fig,axs=plt.subplots(1,2,figsize=(13,5)); bands=["≤250 m","250–500 m","500–1000 m",">1000 m"]; x=np.arange(len(bands)); width=.18
        for ax,m,title in [(axs[0],"visits","Mean visits per tap"),(axs[1],"exposures","Mean tap-water exposures per tap")]:
            for j,sc in enumerate(COOCC_ORDER):
                g=band[band.scenario.eq(sc)].groupby("activity_distance_band",observed=True)[m].agg(["mean","std"]).reindex(bands)
                ax.bar(x+(j-1.5)*width,g["mean"],width,yerr=g["std"],capsize=2,label=COOCC_LABELS[sc])
            ax.set_xticks(x); ax.set_xticklabels(bands); ax.set_title(title); ax.set_xlabel("Distance to nearest activity feature")
        axs[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_tap_distance_band_burden_exposure_mean_sd.png",dpi=260,bbox_inches="tight"); plt.close(fig)

    # 7) Zone-layer shares and concentration: the direct co-occurrence story.
    zl=tables.get("zone_layers",pd.DataFrame()).copy()
    if not zl.empty:
        top=zl.groupby(["scenario","seed","layer"])["share_pct"].max().reset_index(name="top_zone_share_pct")
        gm=top.groupby(["scenario","layer"])["top_zone_share_pct"].mean().unstack().reindex(COOCC_ORDER)
        gs=top.groupby(["scenario","layer"])["top_zone_share_pct"].std().unstack().reindex(index=gm.index,columns=gm.columns).fillna(0)
        gm.index=[COOCC_LABELS[s] for s in gm.index]; gs.index=gm.index
        _annotated_heatmap(gm,gs,"Concentration of each layer in its leading zone",COOCC_OUT/f"{CITY_SLUG}_zone_layer_top_share_mean_sd.png",cbar_label="Top-zone share (%)")
        # Mean share by fixed zone, one panel per scenario.
        layers=["homes","activity","wash","inbound_visits","contaminated_tap_visits","exposures"]
        fig,axs=plt.subplots(1,4,figsize=(20,6),sharey=True)
        vmax=0; mats={}
        for sc in COOCC_ORDER:
            p=zl[zl.scenario.eq(sc)].groupby(["layer","zone_id"])["share_pct"].mean().unstack().reindex(index=layers).fillna(0); mats[sc]=p; vmax=max(vmax,float(p.to_numpy().max()))
        for ax,sc in zip(axs,COOCC_ORDER):
            p=mats[sc]; im=ax.imshow(p.to_numpy(),aspect="auto",cmap="viridis",vmin=0,vmax=max(vmax,1e-9)); ax.set_title(COOCC_LABELS[sc]); ax.set_xticks(range(len(p.columns))); ax.set_xticklabels(p.columns,rotation=90,fontsize=7); ax.set_yticks(range(len(p.index))); ax.set_yticklabels(p.index); ax.set_xlabel("Fixed zone")
        fig.colorbar(im,ax=axs.tolist(),fraction=.02,pad=.02,label="Mean share across seeds (%)"); fig.suptitle("Where urban features, realised use and exposure co-occur"); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_zone_layer_share_story.png",dpi=270,bbox_inches="tight"); plt.close(fig)

    # 8) All disease/mobility time series generated by the ensemble.
    ts=tables.get("timeseries",pd.DataFrame()).copy()
    series_specs=[("incidence_pct","Hourly S→E incidence","% of agents per tick"),("foi","Force of infection","FOI"),("active_exposed_infectious_pct","Active exposed + infectious","% of agents")]
    for metric,title,ylabel in series_specs:
        z=ts[ts.metric.eq(metric)]
        if z.empty: continue
        fig,ax=plt.subplots(figsize=(10,6))
        for sc in COOCC_ORDER:
            g=z[z.scenario.eq(sc)].groupby("tick")["value"].agg(["mean","std"]); ax.plot(g.index,g["mean"],label=COOCC_LABELS[sc]); ax.fill_between(g.index,g["mean"]-g["std"].fillna(0),g["mean"]+g["std"].fillna(0),alpha=.18)
        ax.set_title(f"{title}: mean ± SD across seeds"); ax.set_xlabel("Tick"); ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_{metric}_mean_sd.png",dpi=250,bbox_inches="tight"); plt.close(fig)

    # 9) Every compact disease outcome, not only the original three panels.
    outcomes=tables.get("outcomes",pd.DataFrame()).copy()
    if not outcomes.empty:
        metrics=[c for c in ["cumulative_exposure_attack_rate_pct","peak_6hr_incidence_pct","peak_active_infectious_pct","foi_peak","foi_mean"] if c in outcomes.columns]
        fig,axs=plt.subplots(1,len(metrics),figsize=(5*len(metrics),4.8),squeeze=False)
        for ax,m in zip(axs[0],metrics):
            g=outcomes.groupby("scenario")[m].agg(["mean","std"]).reindex(COOCC_ORDER).fillna(0); x=np.arange(len(g)); bars=ax.bar(x,g["mean"],yerr=g["std"],capsize=3); ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[s] for s in g.index],rotation=28,ha="right"); ax.set_title(m.replace("_"," ")); ax.set_ylabel("Mean ± SD")
            for b,v in zip(bars,g["mean"]): ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{v:.2f}",ha="center",va="bottom",fontsize=7)
        fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_all_outcomes_mean_sd.png",dpi=260,bbox_inches="tight"); plt.close(fig)

    # 10) Exposure-location completeness, so unknown sources are transparent.
    exsrc=tables.get("exposure_source",pd.DataFrame()).copy()
    if not exsrc.empty:
        # Existing exposure-source plot already covers shares; no extra CSV-only metric remains hidden.
        pass



# ===================== PAPER-READY CO-OCCURRENCE TABLES AND FIGURES =====================
# These helpers add analysis/plotting outputs only. They do not change the ABM,
# intervention scenarios, movement rules, epidemiological parameters, or event logging.

def _paper_seed_suffix():
    n = int(max(1, COOCCURRENCE_N_SEEDS))
    return f"pilot_{n}seeds" if n < 10 else f"{n}seeds"


def _paper_path(stem):
    return PAPER_FIG_DIR / f"{CITY_SLUG}_{stem}_{_paper_seed_suffix()}.png"


def _paper_csv_path(stem):
    return PAPER_FIG_DIR / f"{CITY_SLUG}_{stem}_{_paper_seed_suffix()}_data.csv"


def _paper_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
    })


def _panel_label(ax, label):
    ax.text(-0.04, 1.04, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="right")


def _pretty_feature_label(x):
    return PAPER_FEATURE_LABELS.get(str(x), str(x).replace("_", " ").title())


def _pretty_pair_label(pair):
    if isinstance(pair, str):
        return (pair.replace("school_near_tap", "School-near taps")
                    .replace("market_near_tap", "Market-near taps")
                    .replace("home_near_tap", "Home-near taps")
                    .replace("activity_near_tap", "Activity-near taps")
                    .replace("school_near_toilet", "School-near toilets")
                    .replace("market_near_toilet", "Market-near toilets")
                    .replace("home_near_toilet", "Home-near toilets")
                    .replace("activity_near_toilet", "Activity-near toilets")
                    .replace("school_near_market", "School-near markets")
                    .replace("_", " "))
    return str(pair)


def _mean_sd_table(df, index_cols, value_col, col="scenario", order=None):
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    if order is None:
        order = COOCC_ORDER
    group_cols = list(index_cols) + [col]
    agg = d.groupby(group_cols, dropna=False)[value_col].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    if len(index_cols) == 0:
        mean = agg.set_index(col)["mean"].reindex(order).to_frame().T
        sd = agg.set_index(col)["std"].reindex(order).to_frame().T
    else:
        idx = index_cols[0] if len(index_cols) == 1 else index_cols
        mean = agg.pivot_table(index=idx, columns=col, values="mean", aggfunc="first").reindex(columns=order)
        sd = agg.pivot_table(index=idx, columns=col, values="std", aggfunc="first").reindex(index=mean.index, columns=mean.columns).fillna(0)
    mean.columns = [COOCC_LABELS.get(c, str(c)) for c in mean.columns]
    sd.columns = mean.columns
    return mean, sd, agg


def _save_panel_data(stem, data):
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(_paper_csv_path(stem), index=False)
    except Exception as e:
        print(f"[paper plots] could not save data for {stem}: {e}")


def _heatmap_mean_sd_ax(ax, mean, sd=None, title="", cbar_label="", cmap="viridis", fmt=".1f", show_cbar=True):
    if mean is None or mean.empty:
        ax.axis("off"); ax.set_title(title + "\n(no data)"); return None
    arr = mean.to_numpy(dtype=float)
    if sd is None or getattr(sd, "empty", True):
        sd_arr = np.zeros_like(arr)
    else:
        sd_arr = sd.reindex(index=mean.index, columns=mean.columns).fillna(0).to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    vmin = min(0.0, float(np.nanmin(finite))) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mean.columns)))
    ax.set_xticklabels(list(mean.columns), rotation=35, ha="right")
    ax.set_yticks(range(len(mean.index)))
    ax.set_yticklabels([str(i) for i in mean.index])
    threshold = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if not np.isfinite(val):
                txt = "NA"
            else:
                if sd is None:
                    txt = f"{val:{fmt}}"
                else:
                    txt = f"{val:{fmt}}\n±{sd_arr[i,j]:{fmt}}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(val) and val > threshold else "black")
    ax.set_title(title)
    if show_cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cb.set_label(cbar_label)
    return im


def _bar_mean_sd_ax(ax, df, metric, title, ylabel, scenario_col="scenario"):
    if df is None or df.empty or metric not in df.columns:
        ax.axis("off"); ax.set_title(title + "\n(no data)"); return
    g = df.groupby(scenario_col)[metric].agg(["mean", "std"]).reindex(COOCC_ORDER).fillna(0)
    x = np.arange(len(g))
    bars = ax.bar(x, g["mean"], yerr=g["std"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([COOCC_LABELS.get(s, str(s)) for s in g.index], rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for b, v, sd in zip(bars, g["mean"], g["std"]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.1f}\n±{sd:.1f}", ha="center", va="bottom", fontsize=7)


def _feature_pair_burden_rows(result, scenario, seed, radius_m=500.0):
    """Attach realised visit/exposure burden to physical feature-pair categories.

    For tap pairs, exposure is direct tap-water exposure. For toilet and school-market
    pairs, exposure fields are left as missing because those facilities are not direct
    cholera exposure sources in the current model.
    """
    rows = []
    visits = result.get("completed_visits", pd.DataFrame())
    exps = result.get("exposures", pd.DataFrame())
    tap_level = pd.DataFrame(_tap_level_rows(result, scenario, seed))

    def _summ_from_facilities(pair, target_type, target_gdf, mask, nearest_vals=None, exposure_mode="none"):
        if target_gdf is None or getattr(target_gdf, "empty", True):
            rows.append({"scenario":scenario,"seed":seed,"pair":pair,"radius_m":radius_m,"target_facility_type":target_type,
                         "n_target_facilities":0,"pct_target_facilities_in_pair":np.nan,
                         "mean_visits_per_facility":np.nan,"mean_unique_visitors_per_facility":np.nan,
                         "mean_tap_water_exposures_per_facility":np.nan,"mean_exposures_per_1000_visits":np.nan,
                         "median_nearest_pair_distance_m":np.nan})
            return
        t = target_gdf.copy().reset_index(drop=True)
        if "feature_id" not in t.columns:
            t["feature_id"] = [getattr(g, "wkb", i) for i, g in enumerate(t.geometry)]
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.size != len(t):
            mask_arr = np.zeros(len(t), dtype=bool)
        fids = set(t.loc[mask_arr, "feature_id"].dropna().tolist())
        v = visits[(visits["facility_type"].eq(target_type)) & (visits["facility_id"].isin(fids))] if visits is not None and not visits.empty and fids else pd.DataFrame()
        per_fac_vis = v.groupby("facility_id").size() if not v.empty else pd.Series(dtype=float)
        per_fac_users = v.groupby("facility_id")["agent"].nunique() if not v.empty else pd.Series(dtype=float)
        if exposure_mode == "tap" and exps is not None and not exps.empty and fids:
            e = exps[exps.get("exposure_facility_id", pd.Series(index=exps.index, dtype=object)).isin(fids)]
            per_fac_exp = e.groupby("exposure_facility_id").size() if not e.empty else pd.Series(dtype=float)
        else:
            per_fac_exp = pd.Series(dtype=float)
        nfac = int(mask_arr.sum())
        total_vis = float(per_fac_vis.sum())
        total_exp = float(per_fac_exp.sum())
        rows.append({
            "scenario":scenario,"seed":seed,"pair":pair,"radius_m":float(radius_m),"target_facility_type":target_type,
            "n_target_facilities":nfac,"pct_target_facilities_in_pair":100.0*float(nfac)/max(len(t),1),
            "mean_visits_per_facility":float(total_vis/max(nfac,1)) if nfac else np.nan,
            "mean_unique_visitors_per_facility":float(per_fac_users.sum()/max(nfac,1)) if nfac else np.nan,
            "mean_tap_water_exposures_per_facility":float(total_exp/max(nfac,1)) if (nfac and exposure_mode=="tap") else np.nan,
            "mean_exposures_per_1000_visits":float(1000.0*total_exp/max(total_vis,1.0)) if (nfac and exposure_mode=="tap") else np.nan,
            "median_nearest_pair_distance_m":float(np.nanmedian(nearest_vals[mask_arr])) if nearest_vals is not None and np.any(mask_arr) else np.nan,
        })

    # Tap-side pair burden uses the tap-level table already enriched with proximity and exposure counts.
    taps = _active_layer(result, "tap")
    if not tap_level.empty and not taps.empty:
        tap_level = tap_level.copy()
        tap_level["activity_near"] = (pd.to_numeric(tap_level.get("activity_within_500m", 0), errors="coerce").fillna(0) > 0) | (pd.to_numeric(tap_level.get("generic_points_within_500m", 0), errors="coerce").fillna(0) > 0)
        t = taps.copy().reset_index(drop=True)
        # Reorder tap_level to active tap order by feature_id where possible.
        if "feature_id" in t.columns and "tap_id" in tap_level.columns:
            tl = tap_level.set_index("tap_id").reindex(t["feature_id"].tolist()).reset_index(drop=True)
        else:
            tl = tap_level.reset_index(drop=True)
        _summ_from_facilities("school_near_tap", "tap", t, pd.to_numeric(tl.get("schools_within_500m",0), errors="coerce").fillna(0).to_numpy()>0, pd.to_numeric(tl.get("nearest_school_m",np.nan), errors="coerce").to_numpy(), "tap")
        _summ_from_facilities("market_near_tap", "tap", t, pd.to_numeric(tl.get("markets_within_500m",0), errors="coerce").fillna(0).to_numpy()>0, pd.to_numeric(tl.get("nearest_market_m",np.nan), errors="coerce").to_numpy(), "tap")
        _summ_from_facilities("home_near_tap", "tap", t, pd.to_numeric(tl.get("homes_within_500m",0), errors="coerce").fillna(0).to_numpy()>0, pd.to_numeric(tl.get("nearest_home_m",np.nan), errors="coerce").to_numpy(), "tap")
        _summ_from_facilities("activity_near_tap", "tap", t, tl.get("activity_near", pd.Series(False, index=tl.index)).fillna(False).to_numpy(dtype=bool), pd.to_numeric(tl.get("nearest_activity_m",np.nan), errors="coerce").to_numpy(), "tap")

    # Toilet-side pair burden: visit/contact burden only, not direct cholera exposure.
    toilets = _active_layer(result, "toilet")
    if toilets is not None and not toilets.empty:
        tc = _coords_from_gdf(toilets)
        for ref_name, ref_layer, pair in [("school", _active_layer(result,"school"), "school_near_toilet"),
                                          ("market", _active_layer(result,"market"), "market_near_toilet"),
                                          ("home", _unique_home_gdf(result.get("initial_agents", pd.DataFrame())), "home_near_toilet")]:
            rc = _coords_from_gdf(ref_layer)
            if len(rc) and len(tc):
                tr = cKDTree(rc); counts = np.asarray([len(x) for x in tr.query_ball_point(tc, r=radius_m)], dtype=int); dist,_ = tr.query(tc, k=1)
            else:
                counts = np.zeros(len(tc), dtype=int); dist = np.full(len(tc), np.nan)
            _summ_from_facilities(pair, "toilet", toilets, counts>0, dist, "none")
        # activity-near toilet = near school/market/generic activity
        act = []
        for ft in ["school","market","generic_activity"]:
            c = _coords_from_gdf(_active_layer(result, ft))
            if len(c): act.extend(c.tolist())
        if len(act) and len(tc):
            tr = cKDTree(np.asarray(act)); counts = np.asarray([len(x) for x in tr.query_ball_point(tc, r=radius_m)], dtype=int); dist,_ = tr.query(tc, k=1)
        else:
            counts = np.zeros(len(tc), dtype=int); dist = np.full(len(tc), np.nan)
        _summ_from_facilities("activity_near_toilet", "toilet", toilets, counts>0, dist, "none")

    # School-market pair: use market visit burden near schools, but no direct infection assignment.
    markets = _active_layer(result,"market"); schools = _active_layer(result,"school")
    mc, sc = _coords_from_gdf(markets), _coords_from_gdf(schools)
    if len(mc) and len(sc):
        tr = cKDTree(sc); counts = np.asarray([len(x) for x in tr.query_ball_point(mc, r=radius_m)], dtype=int); dist,_ = tr.query(mc,k=1)
        _summ_from_facilities("school_near_market", "market", markets, counts>0, dist, "none")
    return rows


def _tap_typology_rows(result, scenario, seed):
    tap = pd.DataFrame(_tap_level_rows(result, scenario, seed))
    if tap.empty:
        return []
    tap = tap.copy()
    tap["residential_score"] = pd.to_numeric(tap.get("homes_within_500m",0), errors="coerce").fillna(0)
    tap["activity_score"] = pd.to_numeric(tap.get("activity_within_500m",0), errors="coerce").fillna(0)
    rmed = float(tap["residential_score"].median()) if len(tap) else 0.0
    amed = float(tap["activity_score"].median()) if len(tap) else 0.0
    tap["res_high"] = tap["residential_score"] > rmed if rmed > 0 else tap["residential_score"] > 0
    tap["act_high"] = tap["activity_score"] > amed if amed > 0 else tap["activity_score"] > 0
    def _lab(row):
        return ("High residential" if row["res_high"] else "Low residential") + " / " + ("high activity" if row["act_high"] else "low activity")
    tap["tap_typology"] = tap.apply(_lab, axis=1)
    rows=[]
    for typ,d in tap.groupby("tap_typology", dropna=False):
        rows.append({
            "scenario":scenario,"seed":seed,"tap_typology":typ,"n_taps":int(len(d)),
            "mean_visits_per_tap":float(pd.to_numeric(d["visits"], errors="coerce").mean()) if len(d) else np.nan,
            "mean_unique_visitors_per_tap":float(pd.to_numeric(d["unique_visitors"], errors="coerce").mean()) if len(d) else np.nan,
            "mean_tap_water_exposures_per_tap":float(pd.to_numeric(d["tap_water_exposures"], errors="coerce").mean()) if len(d) else np.nan,
            "mean_exposures_per_1000_visits":float(pd.to_numeric(d["tap_water_exposures_per_1000_visits"], errors="coerce").mean()) if len(d) else np.nan,
        })
    return rows


def _activity_facility_cohort_rows(result, scenario, seed, radius_m=500.0):
    visits = result.get("completed_visits", pd.DataFrame())
    init = result.get("initial_agents", pd.DataFrame()).rename(columns={"agent_id":"agent"}).copy()
    exp = result.get("exposures", pd.DataFrame()).copy()
    if init.empty:
        return []
    if exp is not None and not exp.empty:
        first = exp.sort_values("tick").drop_duplicates("agent", keep="first")[["agent","tick","day"]].rename(columns={"tick":"exposure_tick","day":"exposure_day"})
    else:
        first = pd.DataFrame(columns=["agent","exposure_tick","exposure_day"])
    init = init.merge(first, on="agent", how="left")
    init["exposed_by_day2"] = init["exposure_day"].le(2)
    init["exposed_by_day3"] = init["exposure_day"].le(3)
    init["time_to_exposure_days"] = pd.to_numeric(init["exposure_tick"], errors="coerce") / float(TICKS_PER_DAY)
    taps_c = _coords_from_gdf(_active_layer(result,"tap")); toilets_c = _coords_from_gdf(_active_layer(result,"toilet"))
    trees = {"tap": cKDTree(taps_c) if len(taps_c) else None, "toilet": cKDTree(toilets_c) if len(toilets_c) else None}
    rows=[]
    for ft in ["school","market"]:
        g = _active_layer(result, ft).copy().reset_index(drop=True)
        if g.empty:
            continue
        if "feature_id" not in g.columns:
            g["feature_id"] = [getattr(geom,"wkb",i) for i,geom in enumerate(g.geometry)]
        gc = _coords_from_gdf(g)
        for prox_type in ["tap", "toilet"]:
            tree = trees[prox_type]
            if tree is None or len(gc)==0:
                has = np.zeros(len(g), dtype=bool)
            else:
                has = np.asarray([len(x)>0 for x in tree.query_ball_point(gc, r=radius_m)], dtype=bool)
            for prox_label, mask in [(f"{prox_type} within {int(radius_m)} m", has), (f"no {prox_type} within {int(radius_m)} m", ~has)]:
                fids = set(g.loc[mask, "feature_id"].dropna().tolist())
                v = visits[(visits["facility_type"].eq(ft)) & (visits["facility_id"].isin(fids))] if visits is not None and not visits.empty and fids else pd.DataFrame()
                visitors = set(pd.to_numeric(v.get("agent", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)) if not v.empty else set()
                cohort = init[init["agent"].isin(visitors) & init["initial_state"].eq("S")].copy()
                rows.append({
                    "scenario":scenario,"seed":seed,"facility_type":ft,"proximity_class":prox_label,"radius_m":float(radius_m),
                    "n_facilities":int(mask.sum()),"n_visitors":int(len(visitors)),"n_initially_susceptible_visitors":int(len(cohort)),
                    "exposed_by_day2_pct":100.0*float(cohort["exposed_by_day2"].mean()) if len(cohort) else np.nan,
                    "exposed_by_day3_pct":100.0*float(cohort["exposed_by_day3"].mean()) if len(cohort) else np.nan,
                    "median_time_to_exposure_days":float(cohort.loc[cohort["exposure_tick"].notna(),"time_to_exposure_days"].median()) if len(cohort) and cohort["exposure_tick"].notna().any() else np.nan,
                    "mean_visits_per_facility":float(len(v)/max(int(mask.sum()),1)) if int(mask.sum()) else np.nan,
                })
    return rows


def _paper_summary_rows(result, scenario, seed, local=None):
    rows=[]
    try:
        oo,_ = _core_outcome_and_timeseries(result, scenario, seed)
        for k,v in oo.items():
            if k not in ("scenario","seed") and isinstance(v, (int,float,np.integer,np.floating)):
                rows.append({"scenario":scenario,"seed":seed,"metric":k,"value":float(v)})
    except Exception:
        pass
    if local and local.get("zone_layers"):
        z = pd.DataFrame(local["zone_layers"])
        for layer in ["inbound_visits","contaminated_tap_visits","exposures"]:
            dz = z[z["layer"].eq(layer)]
            if not dz.empty:
                rows.append({"scenario":scenario,"seed":seed,"metric":f"top_zone_{layer}_share_pct","value":float(dz["share_pct"].max())})
    return rows


def _draw_paper_layout_map(ax, res, sc, show_legend=False):
    from matplotlib.lines import Line2D
    ax.set_title(COOCC_LABELS.get(sc, sc), fontsize=9)
    zones = res.get("fixed_analysis_zones", None)
    if zones is None or getattr(zones, "empty", True):
        zones = res.get("outer_cells_active", None)
    hubs = res.get("inner_hubs_active", None)
    if zones is not None and not getattr(zones, "empty", True):
        zones.boundary.plot(ax=ax, color="0.75", linewidth=0.5, alpha=0.9, zorder=1)
    if hubs is not None and not getattr(hubs, "empty", True):
        hubs.boundary.plot(ax=ax, color="0.30", linewidth=0.7, alpha=0.7, zorder=2)
    homes = _unique_home_gdf(res.get("initial_agents", pd.DataFrame()))
    if not homes.empty:
        ax.scatter(homes.geometry.x, homes.geometry.y, s=1.2, c="0.70", alpha=0.18, linewidths=0, zorder=0)
    gen = _active_layer(res, "generic_activity")
    if not gen.empty:
        try:
            ax.hexbin(gen.geometry.x, gen.geometry.y, gridsize=34, mincnt=1, alpha=0.18, linewidths=0, cmap="Oranges", zorder=3)
        except Exception:
            gen.plot(ax=ax, color="orange", markersize=2, alpha=0.12, zorder=3)
    styles = {
        "school": ("^", "purple", 22, "Schools"),
        "market": ("s", "green", 20, "Markets"),
        "tap": ("o", "deepskyblue", 24, "Water taps"),
        "toilet": ("x", "black", 18, "Toilets"),
    }
    for ft,(marker,color,ms,label) in styles.items():
        g = _active_layer(res, ft)
        if not g.empty:
            ax.scatter(g.geometry.x, g.geometry.y, marker=marker, s=ms, c=color, alpha=0.85, linewidths=0.6, zorder=4, label=label)
    # Consistent extent using zones/hubs/homes where available.
    ext_g = zones if zones is not None and not getattr(zones,"empty",True) else hubs
    if ext_g is not None and not getattr(ext_g,"empty",True):
        minx,miny,maxx,maxy = ext_g.total_bounds
        pad = 0.03*max(maxx-minx, maxy-miny)
        ax.set_xlim(minx-pad, maxx+pad); ax.set_ylim(miny-pad, maxy+pad)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    if show_legend:
        handles = [Line2D([0],[0],marker="^",color="w",markerfacecolor="purple",label="Schools",markersize=7),
                   Line2D([0],[0],marker="s",color="w",markerfacecolor="green",label="Markets",markersize=7),
                   Line2D([0],[0],marker="o",color="w",markerfacecolor="deepskyblue",label="Water taps",markersize=7),
                   Line2D([0],[0],marker="x",color="black",label="Toilets",markersize=7, linestyle="None")]
        ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=7)


def _plot_paper_spatial_layout_clean(layout_results, stem="paper_fig1A_spatial_layout_clean"):
    if not layout_results:
        return
    fig, axs = plt.subplots(1, len(COOCC_ORDER), figsize=(15, 4.3), squeeze=False)
    for ax, sc in zip(axs[0], COOCC_ORDER):
        if sc in layout_results:
            _draw_paper_layout_map(ax, layout_results[sc], sc, show_legend=(sc==COOCC_ORDER[-1]))
        else:
            ax.axis("off")
    fig.suptitle("Scenario spatial systems: activity and WASH placement")
    fig.tight_layout()
    fig.savefig(_paper_path(stem), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_fig1_spatial_structure(tables, spatial_acc, layout_results):
    _plot_paper_spatial_layout_clean(layout_results)
    pc = tables.get("physical", pd.DataFrame()).copy()
    uf = tables.get("urban_form", pd.DataFrame()).copy()
    fig = plt.figure(figsize=(16, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0])
    map_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, sc in zip(map_axes, COOCC_ORDER):
        if sc in layout_results:
            _draw_paper_layout_map(ax, layout_results[sc], sc, show_legend=(sc==COOCC_ORDER[-1]))
    _panel_label(map_axes[0], "A")
    axB = fig.add_subplot(gs[1, 0:2]); axC = fig.add_subplot(gs[1, 2:4])
    if not uf.empty:
        d = uf[uf["layer"].isin(["hub_area","activity_supply","wash_supply","realised_visits"])].copy()
        d["layer_label"] = d["layer"].map({"hub_area":"Hub area","activity_supply":"Activity supply","wash_supply":"WASH supply","realised_visits":"Realised visits"}).fillna(d["layer"])
        m,s,agg = _mean_sd_table(d, ["layer_label"], "top1_share_pct")
        _save_panel_data("paper_fig1B_top1_share", agg)
        _heatmap_mean_sd_ax(axB, m, s, "Largest-hub share by layer", "Top share (%)", cmap="YlGnBu", fmt=".1f")
        _panel_label(axB, "B")
        figB, ax = plt.subplots(figsize=(8,4.5)); _heatmap_mean_sd_ax(ax, m, s, "Largest-hub share by layer", "Top share (%)", cmap="YlGnBu", fmt=".1f"); figB.tight_layout(); figB.savefig(_paper_path("paper_fig1B_top1_share"), dpi=300, bbox_inches="tight"); plt.close(figB)
    if not pc.empty:
        d = pc[np.isclose(pd.to_numeric(pc["radius_m"], errors="coerce"), 500.0)].copy()
        d["pair"] = d["feature_A"].map(_pretty_feature_label) + "–" + d["feature_B"].map(_pretty_feature_label)
        m,s,agg = _mean_sd_table(d, ["pair"], "pct_A_with_B_in_radius")
        _save_panel_data("paper_fig1C_physical_cooccurrence_500m", agg)
        _heatmap_mean_sd_ax(axC, m, s, "Physical co-occurrence within 500 m", "% of A with B", cmap="YlOrBr", fmt=".1f")
        _panel_label(axC, "C")
        figC, ax = plt.subplots(figsize=(8.5,5)); _heatmap_mean_sd_ax(ax, m, s, "Physical co-occurrence within 500 m", "% of A with B", cmap="YlOrBr", fmt=".1f"); figC.tight_layout(); figC.savefig(_paper_path("paper_fig1C_physical_cooccurrence_500m"), dpi=300, bbox_inches="tight"); plt.close(figC)
    fig.suptitle("Figure 1. Urban and environmental co-occurrence structure")
    fig.savefig(_paper_path("paper_fig1_spatial_structure"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_fig2_amenity_use_access_burden(tables):
    at = tables.get("attraction", pd.DataFrame()).copy()
    tap = tables.get("tap_level", pd.DataFrame()).copy()
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    panels = [
        (axs[0,0], at, "visits_per_agent_day", "A", "Amenity visits per agent per day", "Visits / agent / day", "paper_fig2A_amenity_visits"),
        (axs[0,1], at, "median_trip_distance_m", "B", "Median completed-journey distance", "Metres", "paper_fig2B_amenity_distance"),
        (axs[1,0], at, "visits_per_facility", "C", "Visits per facility", "Visits / facility", "paper_fig2C_visits_per_facility"),
    ]
    for ax, df, metric, lab, title, cbar, stem in panels:
        if df is not None and not df.empty:
            d = df.copy(); d["facility_label"] = d["facility_type"].map(_pretty_feature_label)
            cols = [_pretty_feature_label(x) for x in COOCC_FEATURE_ORDER]
            m = d.groupby(["scenario","facility_label"])[metric].mean().unstack().reindex(index=COOCC_ORDER, columns=cols)
            s = d.groupby(["scenario","facility_label"])[metric].std().unstack().reindex(index=m.index, columns=m.columns).fillna(0)
            m.index=[COOCC_LABELS.get(i,i) for i in m.index]; s.index=m.index
            _save_panel_data(stem, d)
            _heatmap_mean_sd_ax(ax, m, s, title, cbar, cmap="viridis", fmt=".1f")
            figi, axi = plt.subplots(figsize=(8,4.5)); _heatmap_mean_sd_ax(axi, m, s, title, cbar, cmap="viridis", fmt=".1f"); figi.tight_layout(); figi.savefig(_paper_path(stem), dpi=300, bbox_inches="tight"); plt.close(figi)
        else:
            ax.axis("off")
        _panel_label(ax, lab)
    ax = axs[1,1]
    if not tap.empty:
        t = tap.groupby(["scenario","seed"]).agg(
            tap_visits_per_tap=("visits","mean"),
            tap_exposures_per_tap=("tap_water_exposures","mean"),
            tap_exposures_per_1000_visits=("tap_water_exposures_per_1000_visits","mean"),
        ).reset_index()
        long = t.melt(id_vars=["scenario","seed"], var_name="metric", value_name="value")
        long["metric_label"] = long["metric"].map({
            "tap_visits_per_tap":"Tap visits / tap",
            "tap_exposures_per_tap":"Tap-water exposures / tap",
            "tap_exposures_per_1000_visits":"Exposures / 1,000 tap visits",
        })
        m,s,agg = _mean_sd_table(long, ["metric_label"], "value")
        _save_panel_data("paper_fig2D_tap_burden", agg)
        _heatmap_mean_sd_ax(ax, m, s, "Tap burden and exposure intensity", "Mean", cmap="magma", fmt=".1f")
        figi, axi = plt.subplots(figsize=(8,3.8)); _heatmap_mean_sd_ax(axi, m, s, "Tap burden and exposure intensity", "Mean", cmap="magma", fmt=".1f"); figi.tight_layout(); figi.savefig(_paper_path("paper_fig2D_tap_burden"), dpi=300, bbox_inches="tight"); plt.close(figi)
    else:
        ax.axis("off")
    _panel_label(ax, "D")
    fig.suptitle("Figure 2. Amenity use, access distance and WASH burden")
    fig.savefig(_paper_path("paper_fig2_amenity_use_access_burden"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_fig3_pair_cooccurrence_burden(tables):
    pc = tables.get("physical", pd.DataFrame()).copy()
    pb = tables.get("pair_burden", pd.DataFrame()).copy()
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    # A physical pair co-occurrence
    if not pc.empty:
        d = pc[np.isclose(pd.to_numeric(pc["radius_m"], errors="coerce"), 500.0)].copy()
        d["pair_label"] = d["feature_A"].map(_pretty_feature_label) + "–" + d["feature_B"].map(_pretty_feature_label)
        m,s,agg = _mean_sd_table(d, ["pair_label"], "pct_A_with_B_in_radius")
        _save_panel_data("paper_fig3A_physical_pair_cooccurrence", agg)
        _heatmap_mean_sd_ax(axs[0], m, s, "A. Physical co-occurrence", "% within 500 m", cmap="YlOrBr", fmt=".1f")
        figi, ax = plt.subplots(figsize=(8,5)); _heatmap_mean_sd_ax(ax, m, s, "Physical pair co-occurrence", "% within 500 m", cmap="YlOrBr", fmt=".1f"); figi.tight_layout(); figi.savefig(_paper_path("paper_fig3A_physical_pair_cooccurrence"), dpi=300, bbox_inches="tight"); plt.close(figi)
    else: axs[0].axis("off")
    # B visit burden
    if not pb.empty:
        d = pb.copy(); d["pair_label"] = d["pair"].map(_pretty_pair_label)
        m,s,agg = _mean_sd_table(d, ["pair_label"], "mean_visits_per_facility")
        _save_panel_data("paper_fig3B_pair_visit_burden", agg)
        _heatmap_mean_sd_ax(axs[1], m, s, "B. Visit burden at paired facilities", "Visits / facility", cmap="PuBuGn", fmt=".1f")
        figi, ax = plt.subplots(figsize=(8,5)); _heatmap_mean_sd_ax(ax, m, s, "Visit burden at paired facilities", "Visits / facility", cmap="PuBuGn", fmt=".1f"); figi.tight_layout(); figi.savefig(_paper_path("paper_fig3B_pair_visit_burden"), dpi=300, bbox_inches="tight"); plt.close(figi)
        dt = d[d["target_facility_type"].eq("tap") & d["mean_exposures_per_1000_visits"].notna()].copy()
        if not dt.empty:
            m,s,agg = _mean_sd_table(dt, ["pair_label"], "mean_exposures_per_1000_visits")
            _save_panel_data("paper_fig3C_pair_exposure_intensity", agg)
            _heatmap_mean_sd_ax(axs[2], m, s, "C. Tap-water exposure intensity", "Exposures / 1,000 visits", cmap="Reds", fmt=".1f")
            figi, ax = plt.subplots(figsize=(8,4.5)); _heatmap_mean_sd_ax(ax, m, s, "Tap-water exposure intensity", "Exposures / 1,000 visits", cmap="Reds", fmt=".1f"); figi.tight_layout(); figi.savefig(_paper_path("paper_fig3C_pair_exposure_intensity"), dpi=300, bbox_inches="tight"); plt.close(figi)
        else:
            axs[2].axis("off")
    else:
        axs[1].axis("off"); axs[2].axis("off")
    fig.suptitle("Figure 3. Feature-pair co-occurrence, realised use and exposure burden")
    fig.savefig(_paper_path("paper_fig3_pair_cooccurrence_burden"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_exposure_density_clean(spatial_acc, layout_results, stem="paper_fig4D_exposure_density_clean"):
    if spatial_acc is None or "exposure" not in spatial_acc:
        return
    xe, ye = spatial_acc["xe"], spatial_acc["ye"]
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    means={}; vmax=0.0
    for sc in COOCC_ORDER:
        arrs = spatial_acc["exposure"].get(sc, [])
        if not arrs: continue
        stack = np.stack(arrs, axis=0); means[sc]=stack.mean(axis=0); vmax=max(vmax, float(means[sc].max()))
    if not means: return
    fig, axs = plt.subplots(1, len(COOCC_ORDER), figsize=(15,4), sharex=True, sharey=True, squeeze=False)
    for ax, sc in zip(axs[0], COOCC_ORDER):
        arr = means.get(sc, np.zeros_like(next(iter(means.values()))))
        im = ax.imshow(arr, origin="lower", extent=extent, cmap="turbo", vmin=0, vmax=max(vmax,1e-9))
        ax.set_title(COOCC_LABELS.get(sc, sc), fontsize=9)
        lr = layout_results.get(sc, {})
        zones = lr.get("fixed_analysis_zones", None)
        if zones is None or getattr(zones, "empty", True):
            zones = lr.get("outer_cells_active", None)
        if zones is not None and not getattr(zones,"empty",True): zones.boundary.plot(ax=ax, color="white", linewidth=0.45, alpha=0.85)
        taps = _active_layer(lr, "tap")
        if not taps.empty: taps.plot(ax=ax, color="cyan", markersize=7, alpha=0.65)
        ax.axis("off")
    fig.colorbar(im, ax=axs[0].tolist(), fraction=0.02, pad=0.02, label="Mean S→E exposure-event density")
    fig.suptitle("Location of S→E exposure events (mean across seeds)")
    fig.savefig(_paper_path(stem), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_fig4_exposure_outcome_mechanism(tables, spatial_acc, layout_results):
    _plot_paper_exposure_density_clean(spatial_acc, layout_results)
    ts = tables.get("timeseries", pd.DataFrame()).copy()
    out = tables.get("outcomes", pd.DataFrame()).copy()
    es = tables.get("exposure_source", pd.DataFrame()).copy()
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[0.95, 1.05])
    axA = fig.add_subplot(gs[0, 0:2]); axB = fig.add_subplot(gs[0, 2]); axC = fig.add_subplot(gs[0, 3])
    map_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    if not ts.empty:
        z = ts[ts["metric"].eq("cumulative_exposure_pct")].copy()
        for sc in COOCC_ORDER:
            g = z[z["scenario"].eq(sc)].groupby("tick")["value"].agg(["mean","std"]).fillna(0)
            if not g.empty:
                axA.plot(g.index, g["mean"], label=COOCC_LABELS.get(sc, sc))
                axA.fill_between(g.index, g["mean"]-g["std"], g["mean"]+g["std"], alpha=0.16)
        axA.set_title("Cumulative exposure attack rate")
        axA.set_xlabel("Tick"); axA.set_ylabel("% of agents"); axA.legend(fontsize=7)
        figi, ax = plt.subplots(figsize=(8,4.5))
        for sc in COOCC_ORDER:
            g = z[z["scenario"].eq(sc)].groupby("tick")["value"].agg(["mean","std"]).fillna(0)
            if not g.empty:
                ax.plot(g.index, g["mean"], label=COOCC_LABELS.get(sc, sc)); ax.fill_between(g.index, g["mean"]-g["std"], g["mean"]+g["std"], alpha=0.16)
        ax.set_title("Cumulative exposure attack rate"); ax.set_xlabel("Tick"); ax.set_ylabel("% of agents"); ax.legend(fontsize=7); figi.tight_layout(); figi.savefig(_paper_path("paper_fig4A_cumulative_exposure_curve"), dpi=300, bbox_inches="tight"); plt.close(figi)
    _panel_label(axA, "A")
    if not out.empty:
        d = out.melt(id_vars=["scenario","seed"], value_vars=[m for m in ["peak_6hr_incidence_pct","peak_active_infectious_pct"] if m in out.columns], var_name="metric", value_name="value")
        d["metric_label"] = d["metric"].map({"peak_6hr_incidence_pct":"Peak 6h incidence", "peak_active_infectious_pct":"Peak infectious prevalence"})
        x = np.arange(len(d["metric_label"].dropna().unique())); width=0.18; labs=list(d["metric_label"].dropna().unique())
        for j,sc in enumerate(COOCC_ORDER):
            g = d[d["scenario"].eq(sc)].groupby("metric_label")["value"].agg(["mean","std"]).reindex(labs).fillna(0)
            axB.bar(x+(j-1.5)*width, g["mean"], width, yerr=g["std"], capsize=2, label=COOCC_LABELS.get(sc, sc))
        axB.set_xticks(x); axB.set_xticklabels(labs, rotation=25, ha="right"); axB.set_ylabel("% of agents"); axB.set_title("Peak burden")
        figi, ax = plt.subplots(figsize=(7,4.5))
        for j,sc in enumerate(COOCC_ORDER):
            g = d[d["scenario"].eq(sc)].groupby("metric_label")["value"].agg(["mean","std"]).reindex(labs).fillna(0)
            ax.bar(x+(j-1.5)*width, g["mean"], width, yerr=g["std"], capsize=2, label=COOCC_LABELS.get(sc, sc))
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=25, ha="right"); ax.set_ylabel("% of agents"); ax.set_title("Peak burden"); ax.legend(fontsize=7); figi.tight_layout(); figi.savefig(_paper_path("paper_fig4B_peak_burden"), dpi=300, bbox_inches="tight"); plt.close(figi)
    _panel_label(axB, "B")
    if not es.empty:
        sources=["tap_water","home_water","household_secondary","unknown"]
        bottom=np.zeros(len(COOCC_ORDER)); x=np.arange(len(COOCC_ORDER))
        for src in sources:
            g=es[es["source"].eq(src)].groupby("scenario")["share_pct"].mean().reindex(COOCC_ORDER).fillna(0)
            axC.bar(x,g.values,bottom=bottom,label=src.replace("_"," "))
            bottom += g.values
        axC.set_xticks(x); axC.set_xticklabels([COOCC_LABELS[s] for s in COOCC_ORDER], rotation=25, ha="right"); axC.set_ylabel("Exposure source share (%)"); axC.set_title("Exposure source")
        axC.legend(fontsize=6)
        figi, ax = plt.subplots(figsize=(7,4.5)); bottom=np.zeros(len(COOCC_ORDER))
        for src in sources:
            g=es[es["source"].eq(src)].groupby("scenario")["share_pct"].mean().reindex(COOCC_ORDER).fillna(0)
            ax.bar(x,g.values,bottom=bottom,label=src.replace("_"," ")); bottom+=g.values
        ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[s] for s in COOCC_ORDER], rotation=25, ha="right"); ax.set_ylabel("Exposure source share (%)"); ax.set_title("Exposure source"); ax.legend(fontsize=7); figi.tight_layout(); figi.savefig(_paper_path("paper_fig4C_exposure_source_share"), dpi=300, bbox_inches="tight"); plt.close(figi)
    _panel_label(axC, "C")
    # Embedded clean exposure maps.
    xe,ye=spatial_acc["xe"],spatial_acc["ye"]; extent=[xe[0],xe[-1],ye[0],ye[-1]]
    means={}; vmax=0
    for sc in COOCC_ORDER:
        arrs=spatial_acc["exposure"].get(sc,[])
        if arrs:
            stack=np.stack(arrs,axis=0); means[sc]=stack.mean(axis=0); vmax=max(vmax,float(means[sc].max()))
    im=None
    for ax,sc in zip(map_axes, COOCC_ORDER):
        arr=means.get(sc, np.zeros((len(ye)-1,len(xe)-1)))
        im=ax.imshow(arr,origin="lower",extent=extent,cmap="turbo",vmin=0,vmax=max(vmax,1e-9))
        lr=layout_results.get(sc,{})
        zones=lr.get("fixed_analysis_zones", None)
        if zones is None or getattr(zones,"empty",True):
            zones=lr.get("outer_cells_active", None)
        if zones is not None and not getattr(zones,"empty",True): zones.boundary.plot(ax=ax,color="white",linewidth=.45,alpha=.8)
        taps=_active_layer(lr,"tap")
        if not taps.empty: taps.plot(ax=ax,color="cyan",markersize=6,alpha=.6)
        ax.set_title(COOCC_LABELS.get(sc,sc), fontsize=8); ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=map_axes, fraction=0.025, pad=0.015, label="Exposure-event density")
    _panel_label(map_axes[0], "D")
    fig.suptitle("Figure 4. Exposure outcomes and mechanism")
    fig.savefig(_paper_path("paper_fig4_exposure_outcome_mechanism"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_tap_typology(tables):
    tt = tables.get("tap_typology", pd.DataFrame()).copy()
    if tt.empty:
        return
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.4), constrained_layout=True)
    specs=[("n_taps","Number of taps","Count","paper_tap_typology_A_counts"),
           ("mean_visits_per_tap","Visits per tap","Visits / tap","paper_tap_typology_B_visits"),
           ("mean_tap_water_exposures_per_tap","Tap-water exposures per tap","Exposures / tap","paper_tap_typology_C_exposures"),
           ("mean_exposures_per_1000_visits","Exposure intensity","Exposures / 1,000 visits","paper_tap_typology_D_rate")]
    for ax,(metric,title,cbar,stem),lab in zip(axs,specs,list("ABCD")):
        m,s,agg = _mean_sd_table(tt, ["tap_typology"], metric)
        _save_panel_data(stem, agg)
        _heatmap_mean_sd_ax(ax, m, s, title, cbar, cmap="YlGnBu", fmt=".1f")
        _panel_label(ax, lab)
        figi, axi = plt.subplots(figsize=(8,4.5)); _heatmap_mean_sd_ax(axi, m, s, title, cbar, cmap="YlGnBu", fmt=".1f"); figi.tight_layout(); figi.savefig(_paper_path(stem), dpi=300, bbox_inches="tight"); plt.close(figi)
    fig.suptitle("Tap placement typology: residential/activity catchment and WASH burden")
    fig.savefig(_paper_path("paper_tap_typology"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_agent_experienced_cooccurrence_risk(tables):
    risk = tables.get("risk", pd.DataFrame()).copy()
    ac = tables.get("activity_facility_cohort", pd.DataFrame()).copy()
    if risk.empty and ac.empty:
        return
    if not risk.empty:
        r = risk.copy()
        # Use day-2 and day-3 early exposure rows only, with tiny cohorts retained in CSV but omitted from main heatmap if too small.
        rmain = r[(r["horizon_day"].isin([2,3]))].copy()
        try:
            nmean = rmain.groupby("risk_cohort")["n_agents"].mean()
            keep = nmean[nmean >= MIN_COHORT_N_FOR_MAIN_PLOT].index
            rmain = rmain[rmain["risk_cohort"].isin(keep)]
        except Exception:
            pass
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
        d2 = rmain[rmain["horizon_day"].eq(2)]
        if not d2.empty:
            m,s,agg=_mean_sd_table(d2,["risk_cohort"],"risk_day2_pct"); _save_panel_data("paper_agent_risk_A_day2", agg); _heatmap_mean_sd_ax(axs[0],m,s,"Early exposure by experienced co-occurrence\n(day 2)","% exposed",cmap="Reds",fmt=".1f")
        d3 = rmain[rmain["horizon_day"].eq(3)]
        if not d3.empty:
            m,s,agg=_mean_sd_table(d3,["risk_cohort"],"risk_day3_pct"); _save_panel_data("paper_agent_risk_B_day3", agg); _heatmap_mean_sd_ax(axs[1],m,s,"Early exposure by experienced co-occurrence\n(day 3)","% exposed",cmap="Reds",fmt=".1f")
        if not d3.empty:
            m,s,agg=_mean_sd_table(d3,["risk_cohort"],"median_time_to_exposure_days"); _save_panel_data("paper_agent_risk_C_time_to_exposure", agg); _heatmap_mean_sd_ax(axs[2],m,s,"Median time to exposure", "Days", cmap="viridis", fmt=".2f")
        for ax,lab in zip(axs,"ABC"): _panel_label(ax,lab)
        fig.suptitle("Agent-level experienced co-occurrence and early exposure")
        fig.savefig(_paper_path("paper_agent_experienced_cooccurrence_risk"), dpi=320, bbox_inches="tight")
        plt.close(fig)
    if not ac.empty:
        fig, axs = plt.subplots(1, 3, figsize=(18,5.8), constrained_layout=True)
        ac = ac.copy(); ac["cohort_label"] = ac["facility_type"].str.title()+" | "+ac["proximity_class"]
        for ax,metric,title,cbar,stem,lab in [
            (axs[0],"exposed_by_day2_pct","Visitors exposed by day 2","% exposed","paper_activity_facility_cohort_A_day2","A"),
            (axs[1],"exposed_by_day3_pct","Visitors exposed by day 3","% exposed","paper_activity_facility_cohort_B_day3","B"),
            (axs[2],"median_time_to_exposure_days","Median time to exposure","Days","paper_activity_facility_cohort_C_time","C")]:
            m,s,agg=_mean_sd_table(ac,["cohort_label"],metric); _save_panel_data(stem,agg); _heatmap_mean_sd_ax(ax,m,s,title,cbar,cmap="Reds" if "exposed" in metric else "viridis",fmt=".1f" if "exposed" in metric else ".2f"); _panel_label(ax,lab)
        fig.suptitle("Activity-facility visitor cohorts by WASH proximity")
        fig.savefig(_paper_path("paper_activity_facility_cohort"), dpi=320, bbox_inches="tight")
        plt.close(fig)


def _save_paper_figures(tables, spatial_acc, layout_results):
    print("[Paper plots] Saving clean paper-level figures and individual subfigures...")
    _paper_style()
    try:
        _plot_paper_fig1_spatial_structure(tables, spatial_acc, layout_results)
    except Exception as e:
        print("[Paper plots] Figure 1 skipped:", e)
    try:
        _plot_paper_fig2_amenity_use_access_burden(tables)
    except Exception as e:
        print("[Paper plots] Figure 2 skipped:", e)
    try:
        _plot_paper_fig3_pair_cooccurrence_burden(tables)
    except Exception as e:
        print("[Paper plots] Figure 3 skipped:", e)
    try:
        _plot_paper_fig4_exposure_outcome_mechanism(tables, spatial_acc, layout_results)
    except Exception as e:
        print("[Paper plots] Figure 4 skipped:", e)
    try:
        _plot_paper_tap_typology(tables)
    except Exception as e:
        print("[Paper plots] Tap typology skipped:", e)
    try:
        _plot_paper_agent_experienced_cooccurrence_risk(tables)
    except Exception as e:
        print("[Paper plots] Agent/cohort risk skipped:", e)
    print(f"[Paper plots] complete → {PAPER_FIG_DIR}")



# ===================== FINAL CO-OCCURRENCE REFINEMENTS =====================
# This block refines diagnostics and paper figures only. It deliberately does
# not alter the ABM, mobility, intervention, exposure, contamination, or home
# placement logic.

# One scenario order/label/colour convention is used throughout all refined
# figures. Scenarios are consistently shown on the x-axis or as heatmap columns.
SCENARIO_COLOURS = {
    "baseline": "#4C4C4C",
    "intervD_destinations_only": "#E69F00",
    "intervD_facilities_only": "#0072B2",
    "intervD_area_redistribute": "#009E73",
}
COHORT_LABELS = {
    "activity_to_tap_within_6h": "Activity visit followed by tap visit within 6 h",
    "activity_tap_same_day": "Activity and tap visits on the same day",
    "school_tap_same_day": "School and tap visits on the same day",
    "market_tap_same_day": "Market and tap visits on the same day",
    "activity_wash_same_day": "Activity and WASH visits on the same day",
}
PAIR_BURDEN_LABELS = {
    "school_near_tap": "Taps within 500 m of schools",
    "market_near_tap": "Taps within 500 m of markets",
    "home_near_tap": "Taps near residential homes",
    "activity_near_tap": "Taps near activity destinations",
    "school_near_toilet": "Toilets within 500 m of schools",
    "market_near_toilet": "Toilets within 500 m of markets",
    "home_near_toilet": "Toilets near residential homes",
    "activity_near_toilet": "Toilets near activity destinations",
    "school_near_market": "Markets within 500 m of schools",
}

PAPER_MAIN_DIR = PAPER_FIG_DIR / "main"
PAPER_SUPP_DIR = PAPER_FIG_DIR / "supplement"
PAPER_DIAG_DIR = PAPER_FIG_DIR / "diagnostic"
for _d in (PAPER_MAIN_DIR, PAPER_SUPP_DIR, PAPER_DIAG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _seed_note():
    n = int(max(1, COOCCURRENCE_N_SEEDS))
    return f"Mean ± SD across {n} seed{'s' if n != 1 else ''}"


def _paper_path_refined(stem, section="main"):
    base = {"main": PAPER_MAIN_DIR, "supplement": PAPER_SUPP_DIR, "diagnostic": PAPER_DIAG_DIR}.get(section, PAPER_FIG_DIR)
    return base / f"{CITY_SLUG}_{stem}_{_paper_seed_suffix()}.png"


def _paper_csv_path_refined(stem, section="main"):
    base = {"main": PAPER_MAIN_DIR, "supplement": PAPER_SUPP_DIR, "diagnostic": PAPER_DIAG_DIR}.get(section, PAPER_FIG_DIR)
    return base / f"{CITY_SLUG}_{stem}_{_paper_seed_suffix()}_data.csv"


def _save_refined_panel_data(stem, data, section="main"):
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(_paper_csv_path_refined(stem, section), index=False)
    except Exception as e:
        print(f"[paper plots] could not save refined panel data for {stem}: {e}")


def _normalise_initial_agents(result):
    init = result.get("initial_agents", pd.DataFrame())
    if init is None or init.empty:
        return pd.DataFrame(columns=["agent", "role", "initial_state"])
    init = init.copy()
    if "agent" not in init.columns and "agent_id" in init.columns:
        init = init.rename(columns={"agent_id": "agent"})
    if "agent" not in init.columns:
        return pd.DataFrame(columns=["agent", "role", "initial_state"])
    init["agent"] = pd.to_numeric(init["agent"], errors="coerce")
    init = init.dropna(subset=["agent"]).copy()
    init["agent"] = init["agent"].astype(int)
    if "role" not in init.columns:
        init["role"] = "unknown"
    if "initial_state" not in init.columns:
        init["initial_state"] = "S"
    return init


def _first_exposure_events(result):
    exp = result.get("exposures", pd.DataFrame())
    if exp is None or exp.empty or "agent" not in exp.columns:
        return pd.DataFrame(columns=["agent", "exposure_tick", "exposure_day", "exposure_source", "exposure_facility_id", "exposure_x", "exposure_y"])
    exp = exp.copy()
    exp["agent"] = pd.to_numeric(exp["agent"], errors="coerce")
    exp["tick"] = pd.to_numeric(exp.get("tick", np.nan), errors="coerce")
    exp = exp.dropna(subset=["agent", "tick"]).sort_values(["agent", "tick"]).drop_duplicates("agent", keep="first").copy()
    if exp.empty:
        return pd.DataFrame(columns=["agent", "exposure_tick", "exposure_day", "exposure_source", "exposure_facility_id", "exposure_x", "exposure_y"])
    exp["agent"] = exp["agent"].astype(int)
    exp["exposure_tick"] = exp["tick"].astype(int)
    if "day" in exp.columns:
        exp["exposure_day"] = pd.to_numeric(exp["day"], errors="coerce")
    else:
        exp["exposure_day"] = (exp["exposure_tick"] // int(TICKS_PER_DAY)) + 1
    exp["exposure_source"] = exp.get("source", pd.Series("unknown", index=exp.index)).fillna("unknown").astype(str)
    keep = ["agent", "exposure_tick", "exposure_day", "exposure_source"]
    for col in ["exposure_facility_id", "exposure_x", "exposure_y"]:
        if col in exp.columns:
            keep.append(col)
    out = exp[keep].copy()
    for col in ["exposure_facility_id", "exposure_x", "exposure_y"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _normalise_completed_visits(result, init=None):
    visits = result.get("completed_visits", pd.DataFrame())
    if visits is None or visits.empty:
        return pd.DataFrame(columns=["agent", "tick", "facility_type", "facility_id", "facility_x", "facility_y", "role"])
    visits = visits.copy()
    for col in ["agent", "tick"]:
        if col not in visits.columns:
            return pd.DataFrame(columns=["agent", "tick", "facility_type", "facility_id", "facility_x", "facility_y", "role"])
        visits[col] = pd.to_numeric(visits[col], errors="coerce")
    visits = visits.dropna(subset=["agent", "tick"]).copy()
    visits["agent"] = visits["agent"].astype(int)
    visits["tick"] = visits["tick"].astype(int)
    if "facility_type" not in visits.columns:
        visits["facility_type"] = "unknown"
    if "facility_id" not in visits.columns:
        visits["facility_id"] = np.nan
    if "role" not in visits.columns:
        visits["role"] = np.nan
    if init is not None and not init.empty:
        role_lookup = init.set_index("agent")["role"]
        visits["role"] = visits["role"].where(visits["role"].notna(), visits["agent"].map(role_lookup))
    visits["role"] = visits["role"].fillna("unknown").astype(str)
    return visits.sort_values(["agent", "tick", "facility_type"]).reset_index(drop=True)


def _pre_first_exposure_visits(visits, first_exposure):
    """Keep visits made no later than each agent's first S→E event.

    This fixes temporal leakage: an agent is not labelled as a school/market
    visitor because of a visit that occurred after they had already become
    exposed. Unexposed agents retain visits through the end of the run.
    """
    if visits is None or visits.empty:
        return visits.copy() if isinstance(visits, pd.DataFrame) else pd.DataFrame()
    out = visits.copy()
    if first_exposure is None or first_exposure.empty:
        return out
    cutoff = first_exposure.set_index("agent")["exposure_tick"]
    out["_first_exposure_tick"] = out["agent"].map(cutoff)
    keep = out["_first_exposure_tick"].isna() | (out["tick"] <= out["_first_exposure_tick"])
    return out.loc[keep].drop(columns=["_first_exposure_tick"])


def _activity_facility_cohort_rows(result, scenario, seed, radius_m=500.0):
    """Role-specific, pre-first-exposure activity cohorts by nearby WASH.

    This replaces the earlier version that used all visits over the seven-day
    run. The old approach had temporal leakage: visits after an agent's first
    exposure could assign them to a school/market cohort retrospectively.
    """
    init = _normalise_initial_agents(result)
    if init.empty:
        return []
    first = _first_exposure_events(result)
    visits = _normalise_completed_visits(result, init)
    visits = _pre_first_exposure_visits(visits, first)
    base = init.merge(first[["agent", "exposure_tick", "exposure_day"]], on="agent", how="left")
    base["exposed_by_day2"] = pd.to_numeric(base["exposure_day"], errors="coerce").le(2)
    base["exposed_by_day3"] = pd.to_numeric(base["exposure_day"], errors="coerce").le(3)
    base["time_to_exposure_days"] = pd.to_numeric(base["exposure_tick"], errors="coerce") / float(TICKS_PER_DAY)

    taps_c = _coords_from_gdf(_active_layer(result, "tap"))
    toilets_c = _coords_from_gdf(_active_layer(result, "toilet"))
    trees = {
        "tap": cKDTree(taps_c) if len(taps_c) else None,
        "toilet": cKDTree(toilets_c) if len(toilets_c) else None,
    }
    target_role = {"school": "student", "market": "trader"}
    rows = []
    for ft in ["school", "market"]:
        g = _active_layer(result, ft).copy().reset_index(drop=True)
        if g.empty:
            continue
        if "feature_id" not in g.columns:
            g["feature_id"] = [getattr(geom, "wkb", i) for i, geom in enumerate(g.geometry)]
        role = target_role[ft]
        gc = _coords_from_gdf(g)
        for prox_type in ["tap", "toilet"]:
            tree = trees[prox_type]
            if tree is None or len(gc) == 0:
                has = np.zeros(len(g), dtype=bool)
            else:
                has = np.asarray([len(x) > 0 for x in tree.query_ball_point(gc, r=radius_m)], dtype=bool)
            for prox_label, mask in [
                (f"{prox_type} within {int(radius_m)} m", has),
                (f"no {prox_type} within {int(radius_m)} m", ~has),
            ]:
                fids = set(g.loc[mask, "feature_id"].dropna().tolist())
                v = visits[
                    visits["facility_type"].eq(ft)
                    & visits["facility_id"].isin(fids)
                    & visits["role"].eq(role)
                ].copy() if fids and not visits.empty else pd.DataFrame()
                visitors = set(pd.to_numeric(v.get("agent", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)) if not v.empty else set()
                cohort = base[
                    base["agent"].isin(visitors)
                    & base["initial_state"].eq("S")
                    & base["role"].eq(role)
                ].copy()
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "facility_type": ft,
                    "role_target": role,
                    "proximity_type": prox_type,
                    "proximity_class": prox_label,
                    "radius_m": float(radius_m),
                    "cohort_definition": "role-specific visits up to first exposure",
                    "n_facilities": int(mask.sum()),
                    "n_pre_exposure_role_visitors": int(len(visitors)),
                    "n_initially_susceptible_visitors": int(len(cohort)),
                    "exposed_by_day2_pct": 100.0 * float(cohort["exposed_by_day2"].mean()) if len(cohort) else np.nan,
                    "exposed_by_day3_pct": 100.0 * float(cohort["exposed_by_day3"].mean()) if len(cohort) else np.nan,
                    "median_time_to_exposure_days": float(cohort.loc[cohort["exposure_tick"].notna(), "time_to_exposure_days"].median()) if cohort["exposure_tick"].notna().any() else np.nan,
                    "mean_visits_per_facility": float(len(v) / max(int(mask.sum()), 1)) if int(mask.sum()) else np.nan,
                })
    return rows


def _school_wash_alignment_rows(result, scenario, seed, radii=(250.0, 500.0)):
    """Facility-level school/market proximity to taps and toilets plus role use."""
    init = _normalise_initial_agents(result)
    first = _first_exposure_events(result)
    visits = _pre_first_exposure_visits(_normalise_completed_visits(result, init), first)
    rows = []
    target_role = {"school": "student", "market": "trader"}
    taps = _coords_from_gdf(_active_layer(result, "tap"))
    toilets = _coords_from_gdf(_active_layer(result, "toilet"))
    tap_tree = cKDTree(taps) if len(taps) else None
    toilet_tree = cKDTree(toilets) if len(toilets) else None
    for ft in ["school", "market"]:
        g = _active_layer(result, ft).copy().reset_index(drop=True)
        if g.empty:
            continue
        if "feature_id" not in g.columns:
            g["feature_id"] = [getattr(geom, "wkb", i) for i, geom in enumerate(g.geometry)]
        role = target_role[ft]
        for _, rr in g.iterrows():
            geom = rr.geometry
            fid = rr["feature_id"]
            if tap_tree is not None:
                tap_dist = float(tap_tree.query([[geom.x, geom.y]], k=1)[0][0])
                tap_counts = {r: int(len(tap_tree.query_ball_point([[geom.x, geom.y]], r=r)[0])) for r in radii}
            else:
                tap_dist = np.nan
                tap_counts = {r: 0 for r in radii}
            if toilet_tree is not None:
                toilet_dist = float(toilet_tree.query([[geom.x, geom.y]], k=1)[0][0])
                toilet_counts = {r: int(len(toilet_tree.query_ball_point([[geom.x, geom.y]], r=r)[0])) for r in radii}
            else:
                toilet_dist = np.nan
                toilet_counts = {r: 0 for r in radii}
            v = visits[(visits["facility_type"].eq(ft)) & (visits["facility_id"].eq(fid)) & (visits["role"].eq(role))].copy()
            visitors = set(pd.to_numeric(v.get("agent", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)) if not v.empty else set()
            n_susc = int(init[(init["agent"].isin(visitors)) & init["initial_state"].eq("S") & init["role"].eq(role)].shape[0])
            rows.append({
                "scenario": scenario, "seed": seed, "facility_type": ft, "role_target": role,
                "facility_id": fid, "facility_x": float(geom.x), "facility_y": float(geom.y),
                "nearest_tap_m": tap_dist, "nearest_toilet_m": toilet_dist,
                "n_taps_within_250m": tap_counts.get(250.0, 0),
                "n_taps_within_500m": tap_counts.get(500.0, 0),
                "n_toilets_within_250m": toilet_counts.get(250.0, 0),
                "n_toilets_within_500m": toilet_counts.get(500.0, 0),
                "n_pre_exposure_role_visitors": int(len(visitors)),
                "n_initially_susceptible_role_visitors": n_susc,
                "mean_visits_per_facility": float(len(v)),
            })
    return rows


def _school_exposure_pathway_rows(result, scenario, seed, radius_m=500.0):
    """Trace student school visits preceding their first exposure event.

    This is diagnostic evidence for the D1 school-cohort question. It does not
    create a causal pathway; it checks whether high student risk is consistent
    with school visits preceding tap-water exposure at a nearby tap.
    """
    init = _normalise_initial_agents(result)
    first = _first_exposure_events(result)
    visits = _normalise_completed_visits(result, init)
    if init.empty or first.empty or visits.empty:
        return []
    base = init[(init["initial_state"].eq("S")) & (init["role"].eq("student"))][["agent", "role"]].copy()
    exp = first.merge(base, on="agent", how="inner")
    if exp.empty:
        return []
    school_visits = visits[(visits["facility_type"].eq("school")) & (visits["role"].eq("student"))].copy()
    if school_visits.empty:
        return []
    merged = school_visits.merge(exp[["agent", "exposure_tick", "exposure_day", "exposure_source", "exposure_facility_id", "exposure_x", "exposure_y"]], on="agent", how="inner")
    merged = merged[merged["tick"] <= merged["exposure_tick"]].copy()
    if merged.empty:
        return []
    last = merged.sort_values(["agent", "tick"]).drop_duplicates("agent", keep="last").copy()
    rows = []
    for _, r in last.iterrows():
        sx, sy = pd.to_numeric(r.get("facility_x", np.nan), errors="coerce"), pd.to_numeric(r.get("facility_y", np.nan), errors="coerce")
        ex, ey = pd.to_numeric(r.get("exposure_x", np.nan), errors="coerce"), pd.to_numeric(r.get("exposure_y", np.nan), errors="coerce")
        is_tap = str(r.get("exposure_source", "unknown")) == "tap_water"
        if is_tap and np.isfinite(sx) and np.isfinite(sy) and np.isfinite(ex) and np.isfinite(ey):
            d = float(np.hypot(float(sx) - float(ex), float(sy) - float(ey)))
        else:
            d = np.nan
        rows.append({
            "scenario": scenario, "seed": seed, "facility_type": "school", "role_target": "student",
            "agent": int(r["agent"]), "last_school_id_before_exposure": r.get("facility_id", np.nan),
            "last_school_visit_tick": int(r["tick"]), "first_exposure_tick": int(r["exposure_tick"]),
            "time_from_school_visit_to_exposure_h": float(int(r["exposure_tick"]) - int(r["tick"])),
            "exposure_source": str(r.get("exposure_source", "unknown")),
            "exposure_tap_id": r.get("exposure_facility_id", np.nan),
            "distance_last_school_to_exposure_location_m": d,
            "exposure_tap_within_500m_of_last_school": bool(is_tap and np.isfinite(d) and d <= float(radius_m)),
        })
    return rows


def _draw_city_outline(ax, res_or_layout, color="white", linewidth=0.8, alpha=0.9, zorder=5):
    """Draw a single city outline, never the internal hub/Voronoi borders."""
    if not isinstance(res_or_layout, dict):
        return
    zones = res_or_layout.get("fixed_analysis_zones", None)
    if zones is None or getattr(zones, "empty", True):
        zones = res_or_layout.get("outer_cells_active", None)
    if zones is None or getattr(zones, "empty", True):
        return
    try:
        outline = gpd.GeoSeries([zones.unary_union], crs=zones.crs)
        outline.boundary.plot(ax=ax, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    except Exception:
        pass


def _draw_paper_layout_map(ax, res, sc, show_legend=False):
    """Clean map: city outline plus features; no inner hub/analysis borders."""
    from matplotlib.lines import Line2D
    ax.set_title(COOCC_LABELS.get(sc, sc), fontsize=9)
    zones = res.get("fixed_analysis_zones", None)
    if zones is None or getattr(zones, "empty", True):
        zones = res.get("outer_cells_active", None)
    homes = _unique_home_gdf(res.get("initial_agents", pd.DataFrame()))
    if not homes.empty:
        ax.scatter(homes.geometry.x, homes.geometry.y, s=1.2, c="0.70", alpha=0.18, linewidths=0, zorder=0)
    gen = _active_layer(res, "generic_activity")
    if not gen.empty:
        try:
            ax.hexbin(gen.geometry.x, gen.geometry.y, gridsize=34, mincnt=1, alpha=0.18, linewidths=0, cmap="Oranges", zorder=2)
        except Exception:
            gen.plot(ax=ax, color="orange", markersize=2, alpha=0.12, zorder=2)
    styles = {
        "school": ("^", "purple", 22, "Schools"),
        "market": ("s", "green", 20, "Markets"),
        "tap": ("o", "deepskyblue", 24, "Water taps"),
        "toilet": ("x", "black", 18, "Toilets"),
    }
    for ft, (marker, color, ms, label) in styles.items():
        g = _active_layer(res, ft)
        if not g.empty:
            ax.scatter(g.geometry.x, g.geometry.y, marker=marker, s=ms, c=color, alpha=0.85, linewidths=0.6, zorder=4, label=label)
    _draw_city_outline(ax, res, color="0.35", linewidth=0.7, alpha=0.85, zorder=3)
    ext_g = zones
    if ext_g is not None and not getattr(ext_g, "empty", True):
        minx, miny, maxx, maxy = ext_g.total_bounds
        pad = 0.03 * max(maxx - minx, maxy - miny)
        ax.set_xlim(minx - pad, maxx + pad); ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    if show_legend:
        handles = [
            Line2D([0],[0], marker="^", color="w", markerfacecolor="purple", label="Schools", markersize=7),
            Line2D([0],[0], marker="s", color="w", markerfacecolor="green", label="Markets", markersize=7),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="deepskyblue", label="Water taps", markersize=7),
            Line2D([0],[0], marker="x", color="black", label="Toilets", markersize=7, linestyle="None"),
        ]
        ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=7)


def _scenario_feature_heatmap_data(df, feature_col, value_col, feature_order=None):
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    agg = d.groupby([feature_col, "scenario"], dropna=False)[value_col].agg(["mean", "std"]).reset_index()
    mean = agg.pivot_table(index=feature_col, columns="scenario", values="mean", aggfunc="first")
    sd = agg.pivot_table(index=feature_col, columns="scenario", values="std", aggfunc="first")
    if feature_order is not None:
        mean = mean.reindex(index=feature_order)
        sd = sd.reindex(index=mean.index)
    mean = mean.reindex(columns=COOCC_ORDER)
    sd = sd.reindex(index=mean.index, columns=mean.columns).fillna(0.0)
    mean.columns = [COOCC_LABELS.get(c, str(c)) for c in mean.columns]
    sd.columns = mean.columns
    return mean, sd, agg


def _plot_paper_fig2_amenity_use_access_burden(tables):
    """Three clean, scenario-column panels; tap-rate heatmap is kept out of main Figure 2."""
    at = tables.get("attraction", pd.DataFrame()).copy()
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    panels = [
        ("visits_per_agent_day", "A", "Visits attracted by each amenity type", "Visits / agent / day", "paper_fig2A_amenity_visits"),
        ("median_trip_distance_m", "B", "Distance travelled to each amenity type", "Median completed journey (m)", "paper_fig2B_amenity_distance"),
        ("visits_per_facility", "C", "Use burden per individual facility", "Visits / facility", "paper_fig2C_visits_per_facility"),
    ]
    for ax, (metric, lab, title, cbar, stem) in zip(axs, panels):
        if at.empty or metric not in at.columns:
            ax.axis("off"); ax.set_title(title + "\n(no data)"); continue
        d = at.copy(); d["facility_label"] = d["facility_type"].map(_pretty_feature_label)
        order = [_pretty_feature_label(x) for x in COOCC_FEATURE_ORDER]
        m, s, agg = _scenario_feature_heatmap_data(d, "facility_label", metric, order)
        _save_refined_panel_data(stem, agg, "main")
        _heatmap_mean_sd_ax(ax, m, s, title, cbar, cmap="viridis", fmt=".1f")
        _panel_label(ax, lab)
        f, a = plt.subplots(figsize=(8.2, 4.8)); _heatmap_mean_sd_ax(a, m, s, title, cbar, cmap="viridis", fmt=".1f"); a.text(0.01, -0.20, _seed_note(), transform=a.transAxes, fontsize=8); f.tight_layout(); f.savefig(_paper_path_refined(stem, "main"), dpi=320, bbox_inches="tight"); plt.close(f)
    fig.suptitle("Figure 2. Which amenity systems attract movement and carry use burden")
    fig.text(0.5, 0.01, _seed_note(), ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("paper_fig2_amenity_use_access", "main"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_fig3_pair_cooccurrence_burden(tables):
    """No duplicated physical co-occurrence: Figure 1 contains the 500 m structure panel."""
    pb = tables.get("pair_burden", pd.DataFrame()).copy()
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.0), constrained_layout=True)
    specs = [
        ("mean_visits_per_facility", "A", "Visits per paired facility", "Visits / facility", "PuBuGn", ".1f", "paper_fig3A_pair_visit_burden"),
        ("mean_tap_water_exposures_per_facility", "B", "Tap-water exposures per paired tap", "Exposures / tap", "OrRd", ".1f", "paper_fig3B_pair_tap_exposure_burden"),
        ("mean_exposures_per_1000_visits", "C", "Tap-water exposure intensity at paired taps", "Exposures / 1,000 visits", "Reds", ".1f", "paper_fig3C_pair_exposure_intensity"),
    ]
    for ax, (metric, lab, title, cbar, cmap, fmt, stem) in zip(axs, specs):
        if pb.empty or metric not in pb.columns:
            ax.axis("off"); ax.set_title(title + "\n(no data)"); continue
        d = pb.copy()
        if metric != "mean_visits_per_facility":
            d = d[d["target_facility_type"].eq("tap")].copy()
        d = d[d[metric].notna()].copy()
        if d.empty:
            ax.axis("off"); ax.set_title(title + "\n(no data)"); continue
        d["pair_label"] = d["pair"].map(lambda x: PAIR_BURDEN_LABELS.get(x, _pretty_pair_label(x)))
        m, s, agg = _scenario_feature_heatmap_data(d, "pair_label", metric)
        _save_refined_panel_data(stem, agg, "main")
        _heatmap_mean_sd_ax(ax, m, s, title, cbar, cmap=cmap, fmt=fmt)
        _panel_label(ax, lab)
        f, a = plt.subplots(figsize=(8.8, 5.3)); _heatmap_mean_sd_ax(a, m, s, title, cbar, cmap=cmap, fmt=fmt); a.text(0.01, -0.20, _seed_note(), transform=a.transAxes, fontsize=8); f.tight_layout(); f.savefig(_paper_path_refined(stem, "main"), dpi=320, bbox_inches="tight"); plt.close(f)
    fig.suptitle("Figure 3. Realised use and exposure burden at paired facilities")
    fig.text(0.5, 0.01, "Physical proximity is shown once in Figure 1; this figure shows what those pairings do in use.", ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("paper_fig3_pair_burden_exposure", "main"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _spatial_mean_field(spatial_acc, field_key):
    if spatial_acc is None or field_key not in spatial_acc:
        return {}, 0.0
    means, vmax = {}, 0.0
    for sc in COOCC_ORDER:
        arrs = spatial_acc.get(field_key, {}).get(sc, [])
        if not arrs:
            continue
        stack = np.stack(arrs, axis=0)
        means[sc] = stack.mean(axis=0)
        vmax = max(vmax, float(np.nanmax(means[sc])) if means[sc].size else 0.0)
    return means, vmax


def _draw_spatial_overlay(ax, layout, overlay="wash"):
    if not isinstance(layout, dict):
        return
    if overlay == "all_pois":
        styles = [("school", "^", "purple", 16, "Schools"), ("market", "s", "green", 14, "Markets")]
        for ft, mk, co, size, _ in styles:
            g = _active_layer(layout, ft)
            if not g.empty:
                ax.scatter(g.geometry.x, g.geometry.y, marker=mk, s=size, c=co, alpha=0.7, linewidths=0.4, zorder=4)
    for ft, mk, co, size, _ in [("tap", "o", "deepskyblue", 16, "Water taps"), ("toilet", "x", "black", 18, "Toilets")]:
        g = _active_layer(layout, ft)
        if not g.empty:
            ax.scatter(g.geometry.x, g.geometry.y, marker=mk, s=size, c=co, alpha=0.8, linewidths=0.6, zorder=5)
    _draw_city_outline(ax, layout, color="white", linewidth=0.8, alpha=0.9, zorder=6)


def _plot_paper_spatial_density_map(spatial_acc, layout_results, field_key="exposure", overlay="wash", stem="paper_map", title=""):
    if spatial_acc is None or field_key not in spatial_acc:
        return
    means, vmax = _spatial_mean_field(spatial_acc, field_key)
    if not means:
        return
    xe, ye = spatial_acc["xe"], spatial_acc["ye"]
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    fig, axs = plt.subplots(1, len(COOCC_ORDER), figsize=(16, 4.4), sharex=True, sharey=True, squeeze=False)
    im = None
    for ax, sc in zip(axs[0], COOCC_ORDER):
        arr = means.get(sc, np.zeros_like(next(iter(means.values()))))
        im = ax.imshow(arr, origin="lower", extent=extent, cmap="turbo", vmin=0, vmax=max(vmax, 1e-9), zorder=1)
        _draw_spatial_overlay(ax, layout_results.get(sc, {}), overlay=overlay)
        ax.set_title(COOCC_LABELS.get(sc, sc), fontsize=9)
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axs[0].tolist(), fraction=0.02, pad=0.02, label="Mean density across seeds")
    fig.suptitle(title)
    fig.text(0.5, 0.01, _seed_note(), ha="center", fontsize=8)
    fig.savefig(_paper_path_refined(stem, "main" if field_key == "exposure" else "supplement"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_exposure_density_clean(spatial_acc, layout_results, stem="paper_fig4C_exposure_density_wash_overlay"):
    _plot_paper_spatial_density_map(
        spatial_acc, layout_results, field_key="exposure", overlay="wash", stem=stem,
        title="Location of S→E exposure events over 7 days: taps and toilets shown"
    )


def _plot_paper_peak_ei_density_clean(spatial_acc, layout_results, stem="supp_peak_EI_density_clean"):
    _plot_paper_spatial_density_map(
        spatial_acc, layout_results, field_key="peak_ei", overlay="all_pois", stem=stem,
        title="Location of exposed and infectious agents at peak burden: all POIs shown"
    )


def _plot_peak_outcomes_by_scenario(ax, out, stem=None):
    metrics = [m for m in ["peak_6hr_incidence_pct", "peak_active_infectious_pct"] if m in out.columns]
    if out is None or out.empty or not metrics:
        ax.axis("off"); ax.set_title("Peak incidence and infectious prevalence\n(no data)"); return
    labels = {
        "peak_6hr_incidence_pct": "Peak 6-hour incidence",
        "peak_active_infectious_pct": "Peak infectious prevalence",
    }
    x = np.arange(len(COOCC_ORDER)); width = 0.34 if len(metrics) == 2 else 0.52
    for j, metric in enumerate(metrics):
        g = out.groupby("scenario")[metric].agg(["mean", "std"]).reindex(COOCC_ORDER).fillna(0.0)
        offset = (j - (len(metrics) - 1) / 2.0) * width
        ax.bar(x + offset, g["mean"], width, yerr=g["std"], capsize=3, label=labels.get(metric, metric))
    ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[s] for s in COOCC_ORDER], rotation=25, ha="right")
    ax.set_ylabel("% of agents")
    ax.set_title("Peak incidence and infectious prevalence")
    ax.legend(fontsize=7, frameon=True)
    if stem:
        f, a = plt.subplots(figsize=(8.5, 4.8)); _plot_peak_outcomes_by_scenario(a, out, stem=None); a.text(0.01, -0.22, _seed_note(), transform=a.transAxes, fontsize=8); f.tight_layout(); f.savefig(_paper_path_refined(stem, "main"), dpi=320, bbox_inches="tight"); plt.close(f)


def _draw_embedded_exposure_maps(fig, gs, row, spatial_acc, layout_results, overlay, panel_label, row_title):
    axes = [fig.add_subplot(gs[row, i]) for i in range(4)]
    means, vmax = _spatial_mean_field(spatial_acc, "exposure")
    xe, ye = spatial_acc["xe"], spatial_acc["ye"]
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    im = None
    for ax, sc in zip(axes, COOCC_ORDER):
        arr = means.get(sc, np.zeros((len(ye)-1, len(xe)-1)))
        im = ax.imshow(arr, origin="lower", extent=extent, cmap="turbo", vmin=0, vmax=max(vmax, 1e-9), zorder=1)
        _draw_spatial_overlay(ax, layout_results.get(sc, {}), overlay=overlay)
        ax.set_title(COOCC_LABELS.get(sc, sc), fontsize=8)
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.022, pad=0.012, label="Mean exposure-event density")
    _panel_label(axes[0], panel_label)
    axes[0].text(0.0, 1.12, row_title, transform=axes[0].transAxes, fontsize=9, fontweight="bold", ha="left")
    return axes


def _plot_paper_fig4_exposure_outcome_mechanism(tables, spatial_acc, layout_results):
    """Main Figure 4 without exposure-source panel; scenarios are on x in bar panel."""
    ts = tables.get("timeseries", pd.DataFrame()).copy()
    out = tables.get("outcomes", pd.DataFrame()).copy()
    fig = plt.figure(figsize=(17, 13.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[0.82, 1.05, 1.05])
    axA = fig.add_subplot(gs[0, 0:2]); axB = fig.add_subplot(gs[0, 2:4])
    if not ts.empty:
        z = ts[ts["metric"].eq("cumulative_exposure_pct")].copy()
        for sc in COOCC_ORDER:
            g = z[z["scenario"].eq(sc)].groupby("tick")["value"].agg(["mean", "std"]).fillna(0.0)
            if not g.empty:
                axA.plot(g.index, g["mean"], label=COOCC_LABELS.get(sc, sc), color=SCENARIO_COLOURS.get(sc))
                axA.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=SCENARIO_COLOURS.get(sc), alpha=0.14)
        axA.set_title("Cumulative exposure attack rate")
        axA.set_xlabel("Simulation tick (hour)"); axA.set_ylabel("% of agents"); axA.legend(fontsize=7, frameon=True)
        f, a = plt.subplots(figsize=(8.5, 4.8))
        for sc in COOCC_ORDER:
            g = z[z["scenario"].eq(sc)].groupby("tick")["value"].agg(["mean", "std"]).fillna(0.0)
            if not g.empty:
                a.plot(g.index, g["mean"], label=COOCC_LABELS.get(sc, sc), color=SCENARIO_COLOURS.get(sc))
                a.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=SCENARIO_COLOURS.get(sc), alpha=0.14)
        a.set_title("Cumulative exposure attack rate"); a.set_xlabel("Simulation tick (hour)"); a.set_ylabel("% of agents"); a.legend(fontsize=7); a.text(0.01, -0.22, _seed_note(), transform=a.transAxes, fontsize=8); f.tight_layout(); f.savefig(_paper_path_refined("paper_fig4A_cumulative_exposure_curve", "main"), dpi=320, bbox_inches="tight"); plt.close(f)
    else:
        axA.axis("off")
    _panel_label(axA, "A")
    _plot_peak_outcomes_by_scenario(axB, out, stem="paper_fig4B_peak_burden")
    _panel_label(axB, "B")
    _draw_embedded_exposure_maps(fig, gs, 1, spatial_acc, layout_results, "wash", "C", "S→E exposure events over 7 days: WASH overlay")
    _draw_embedded_exposure_maps(fig, gs, 2, spatial_acc, layout_results, "all_pois", "D", "S→E exposure events over 7 days: all-POI overlay")
    fig.suptitle("Figure 4. Exposure outcomes and exposure geography")
    fig.text(0.5, 0.005, "Colours show where susceptible agents became exposed, not where infectious agents were located at the peak. " + _seed_note(), ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("paper_fig4_exposure_outcome_mechanism", "main"), dpi=320, bbox_inches="tight")
    plt.close(fig)
    _plot_paper_exposure_density_clean(spatial_acc, layout_results)
    _plot_paper_spatial_density_map(spatial_acc, layout_results, field_key="exposure", overlay="all_pois", stem="paper_fig4D_exposure_density_all_pois_overlay", title="Location of S→E exposure events over 7 days: all activity and WASH POIs shown")
    _plot_paper_peak_ei_density_clean(spatial_acc, layout_results)


def _plot_paper_agent_experienced_cooccurrence_risk(tables):
    """Supplementary temporal-sequence and role-specific cohort figures with clear labels."""
    risk = tables.get("risk", pd.DataFrame()).copy()
    ac = tables.get("activity_facility_cohort", pd.DataFrame()).copy()
    if not risk.empty:
        r = risk.copy()
        r["risk_label"] = r["risk_cohort"].map(lambda x: COHORT_LABELS.get(x, str(x).replace("_", " ").title()))
        rmain = r[r["horizon_day"].isin([2, 3])].copy() if "horizon_day" in r.columns else pd.DataFrame()
        if not rmain.empty:
            try:
                nmean = rmain.groupby("risk_label")["n_agents"].mean()
                keep = nmean[nmean >= MIN_COHORT_N_FOR_MAIN_PLOT].index
                rmain = rmain[rmain["risk_label"].isin(keep)]
            except Exception:
                pass
            fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
            specs = [
                (rmain[rmain["horizon_day"].eq(2)], "risk_day2_pct", "Early exposure by experienced sequence (day 2)", "% exposed", "Reds", ".1f", "A", "supp_agent_risk_A_day2"),
                (rmain[rmain["horizon_day"].eq(3)], "risk_day3_pct", "Early exposure by experienced sequence (day 3)", "% exposed", "Reds", ".1f", "B", "supp_agent_risk_B_day3"),
                (rmain[rmain["horizon_day"].eq(3)], "median_time_to_exposure_days", "Median time to exposure", "Days", "viridis", ".2f", "C", "supp_agent_risk_C_time_to_exposure"),
            ]
            for ax, (d, metric, title, cbar, cmap, fmt, lab, stem) in zip(axs, specs):
                if d.empty or metric not in d.columns:
                    ax.axis("off"); continue
                m, s, agg = _scenario_feature_heatmap_data(d, "risk_label", metric)
                _save_refined_panel_data(stem, agg, "supplement")
                _heatmap_mean_sd_ax(ax, m, s, title, cbar, cmap=cmap, fmt=fmt)
                _panel_label(ax, lab)
            fig.suptitle("Experienced co-occurrence: temporal sequences, not physical distance")
            fig.text(0.5, 0.01, "A sequence means that the same agent visited an activity destination and then a tap within the stated time window. " + _seed_note(), ha="center", fontsize=8)
            fig.savefig(_paper_path_refined("supp_agent_experienced_cooccurrence_risk", "supplement"), dpi=320, bbox_inches="tight")
            plt.close(fig)
    if not ac.empty:
        ac = ac.copy()
        ac["cohort_label"] = ac["facility_type"].str.title() + " visitors | " + ac["proximity_class"]
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
        specs = [
            ("exposed_by_day2_pct", "Visitors exposed by day 2", "% exposed", "Reds", ".1f", "A", "supp_activity_facility_cohort_A_day2"),
            ("exposed_by_day3_pct", "Visitors exposed by day 3", "% exposed", "Reds", ".1f", "B", "supp_activity_facility_cohort_B_day3"),
            ("median_time_to_exposure_days", "Median time to exposure", "Days", "viridis", ".2f", "C", "supp_activity_facility_cohort_C_time"),
        ]
        for ax, (metric, title, cbar, cmap, fmt, lab, stem) in zip(axs, specs):
            m, s, agg = _scenario_feature_heatmap_data(ac, "cohort_label", metric)
            _save_refined_panel_data(stem, agg, "supplement")
            _heatmap_mean_sd_ax(ax, m, s, title, cbar, cmap=cmap, fmt=fmt)
            _panel_label(ax, lab)
        fig.suptitle("Role-specific activity visitor cohorts by nearby WASH")
        fig.text(0.5, 0.01, "Cohort membership uses only visits up to each agent's first exposure event. " + _seed_note(), ha="center", fontsize=8)
        fig.savefig(_paper_path_refined("supp_activity_facility_cohort", "supplement"), dpi=320, bbox_inches="tight")
        plt.close(fig)


def _plot_school_wash_alignment(tables):
    sw = tables.get("school_wash_alignment", pd.DataFrame()).copy()
    if sw.empty:
        return
    sw = sw[sw["facility_type"].eq("school")].copy()
    if sw.empty:
        return
    summary = sw.groupby(["scenario", "seed"], dropna=False).agg(
        pct_schools_tap_500=("n_taps_within_500m", lambda x: 100.0 * float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean()) if len(x) else np.nan),
        pct_schools_toilet_500=("n_toilets_within_500m", lambda x: 100.0 * float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean()) if len(x) else np.nan),
        median_school_tap_m=("nearest_tap_m", "median"),
        median_school_toilet_m=("nearest_toilet_m", "median"),
    ).reset_index()
    _save_refined_panel_data("supp_school_wash_alignment", summary, "supplement")
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    specs = [
        ("pct_schools_tap_500", "% of schools with tap within 500 m", "% of schools"),
        ("pct_schools_toilet_500", "% of schools with toilet within 500 m", "% of schools"),
        ("median_school_tap_m", "Median school-to-nearest-tap distance", "Metres"),
        ("median_school_toilet_m", "Median school-to-nearest-toilet distance", "Metres"),
    ]
    for ax, (metric, title, ylabel) in zip(axs.ravel(), specs):
        _bar_mean_sd_ax(ax, summary, metric, title, ylabel)
    fig.suptitle("School–WASH alignment diagnostic")
    fig.text(0.5, 0.01, "Structural diagnostic: this checks whether D1 moved schools into different proximity to the unchanged WASH layer.", ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("supp_school_wash_alignment", "supplement"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_school_cohort_seed_consistency(tables):
    ac = tables.get("activity_facility_cohort", pd.DataFrame()).copy()
    if ac.empty:
        return
    ac = ac[ac["facility_type"].eq("school")].copy()
    if ac.empty:
        return
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    specs = [
        ("tap", "exposed_by_day2_pct", "Students visiting schools: exposure by day 2", "% exposed"),
        ("toilet", "exposed_by_day2_pct", "Students visiting schools: exposure by day 2", "% exposed"),
        ("tap", "median_time_to_exposure_days", "Students visiting schools: median time to exposure", "Days"),
        ("toilet", "median_time_to_exposure_days", "Students visiting schools: median time to exposure", "Days"),
    ]
    for ax, (prox, metric, title, ylabel) in zip(axs.ravel(), specs):
        d = ac[ac["proximity_type"].eq(prox)].copy()
        classes = [f"{prox} within 500 m", f"no {prox} within 500 m"]
        width = 0.34; x = np.arange(len(COOCC_ORDER))
        for j, cl in enumerate(classes):
            sub = d[d["proximity_class"].eq(cl)].copy()
            g = sub.groupby("scenario")[metric].agg(["mean", "std"]).reindex(COOCC_ORDER)
            offset = (j - 0.5) * width
            bars = ax.bar(x + offset, g["mean"], width, yerr=g["std"].fillna(0), capsize=3, label=cl)
            for k, sc in enumerate(COOCC_ORDER):
                raw = sub[sub["scenario"].eq(sc)][metric].dropna().to_numpy(dtype=float)
                if len(raw):
                    jitter = np.linspace(-0.05, 0.05, len(raw)) if len(raw) > 1 else np.array([0.0])
                    ax.scatter(np.full(len(raw), x[k] + offset) + jitter, raw, s=18, color="black", alpha=0.7, zorder=4)
        ax.set_xticks(x); ax.set_xticklabels([COOCC_LABELS[s] for s in COOCC_ORDER], rotation=25, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title + f"\n({prox} proximity; raw seed points shown)")
        ax.legend(fontsize=7, frameon=True)
    fig.suptitle("D1 school-cohort consistency diagnostic")
    fig.text(0.5, 0.01, "Each dot is one seed. Cohorts include only initially susceptible students with school visits up to first exposure.", ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("supp_school_cohort_seed_consistency", "supplement"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_school_exposure_pathway(tables):
    sp = tables.get("school_exposure_pathway", pd.DataFrame()).copy()
    if sp.empty:
        return
    # One row per student with a school visit before first exposure.
    summary = sp.groupby(["scenario", "seed"], dropna=False).agg(
        pct_tap_water_exposure=("exposure_source", lambda x: 100.0 * float(pd.Series(x).eq("tap_water").mean()) if len(x) else np.nan),
        pct_exposure_tap_near_last_school=("exposure_tap_within_500m_of_last_school", lambda x: 100.0 * float(pd.Series(x).fillna(False).astype(bool).mean()) if len(x) else np.nan),
        median_school_to_exposure_tap_m=("distance_last_school_to_exposure_location_m", "median"),
        median_hours_school_to_exposure=("time_from_school_visit_to_exposure_h", "median"),
        n_student_paths=("agent", "nunique"),
    ).reset_index()
    _save_refined_panel_data("supp_school_exposure_pathway", summary, "supplement")
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    specs = [
        ("pct_tap_water_exposure", "% of student pathways with tap-water exposure", "% of pathways"),
        ("pct_exposure_tap_near_last_school", "% of exposure taps within 500 m of last school", "% of pathways"),
        ("median_school_to_exposure_tap_m", "Median distance: last school to exposure tap", "Metres"),
        ("median_hours_school_to_exposure", "Median time: last school visit to exposure", "Hours"),
    ]
    for ax, (metric, title, ylabel) in zip(axs.ravel(), specs):
        _bar_mean_sd_ax(ax, summary, metric, title, ylabel)
    fig.suptitle("Student school-to-exposure pathway diagnostic")
    fig.text(0.5, 0.01, "Diagnostic only: proximity/sequence consistency, not proof that schools or toilets directly transmit infection.", ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("supp_school_exposure_pathway", "supplement"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_conceptual_counterfactuals():
    """A small methods schematic explaining D1/D2 as analytical component counterfactuals."""
    from matplotlib.patches import FancyBboxPatch
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.8), constrained_layout=True)
    items = [
        ("Baseline clustered", "Activity and WASH\nfollow the clustered\nreference arrangement."),
        ("D1 activity-only", "Activity destinations\nare redistributed.\nWASH remains fixed."),
        ("D2 WASH-only", "Taps and toilets\nare redistributed.\nActivity remains fixed."),
        ("D3 coordinated", "Activity and WASH\nare redistributed\ntogether."),
    ]
    for ax, (title, body) in zip(axs, items):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        box = FancyBboxPatch((0.05, 0.12), 0.90, 0.76, boxstyle="round,pad=0.03", facecolor="white", edgecolor="0.45", linewidth=1.0)
        ax.add_patch(box)
        ax.scatter([0.18, 0.28, 0.38], [0.72, 0.66, 0.77], s=24, color="0.65", alpha=0.7, label="Homes" if title == "Baseline clustered" else None)
        if title in ("Baseline clustered", "D2 WASH-only"):
            ax.scatter([0.45, 0.50, 0.56], [0.52, 0.56, 0.48], marker="^", s=45, color="purple")
            ax.scatter([0.55, 0.60], [0.58, 0.45], marker="s", s=40, color="green")
        else:
            ax.scatter([0.42, 0.70, 0.62], [0.52, 0.60, 0.34], marker="^", s=45, color="purple")
            ax.scatter([0.50, 0.78], [0.58, 0.42], marker="s", s=40, color="green")
        if title in ("Baseline clustered", "D1 activity-only"):
            ax.scatter([0.48, 0.57], [0.30, 0.27], marker="o", s=44, color="deepskyblue")
            ax.scatter([0.63], [0.32], marker="x", s=46, color="black")
        else:
            ax.scatter([0.35, 0.73], [0.31, 0.36], marker="o", s=44, color="deepskyblue")
            ax.scatter([0.58], [0.28], marker="x", s=46, color="black")
        ax.text(0.5, 0.93, title, ha="center", va="center", fontweight="bold", fontsize=10)
        ax.text(0.5, 0.11, body, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Analytical decomposition of urban co-occurrence")
    fig.text(0.5, 0.005, "Homes remain fixed in every scenario. D1 and D2 are component counterfactuals, not literal redevelopment plans; D3 is the coordinated activity + WASH case.", ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("paper_conceptual_component_counterfactuals", "main"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _save_paper_figures(tables, spatial_acc, layout_results):
    print("[Paper plots] Saving refined main, supplementary and diagnostic paper figures...")
    _paper_style()
    functions = [
        ("Conceptual scenario schematic", lambda: _plot_paper_conceptual_counterfactuals()),
        ("Figure 1", lambda: _plot_paper_fig1_spatial_structure(tables, spatial_acc, layout_results)),
        ("Figure 2", lambda: _plot_paper_fig2_amenity_use_access_burden(tables)),
        ("Figure 3", lambda: _plot_paper_fig3_pair_cooccurrence_burden(tables)),
        ("Figure 4", lambda: _plot_paper_fig4_exposure_outcome_mechanism(tables, spatial_acc, layout_results)),
        ("Tap typology", lambda: _plot_paper_tap_typology(tables)),
        ("Experienced/cohort risk", lambda: _plot_paper_agent_experienced_cooccurrence_risk(tables)),
        ("School-WASH alignment", lambda: _plot_school_wash_alignment(tables)),
        ("School cohort seed consistency", lambda: _plot_school_cohort_seed_consistency(tables)),
        ("School exposure pathway", lambda: _plot_school_exposure_pathway(tables)),
    ]
    for name, func in functions:
        try:
            func()
        except Exception as e:
            print(f"[Paper plots] {name} skipped: {e}")
    print(f"[Paper plots] complete → {PAPER_FIG_DIR}")


# ===================== CLEAN FINAL MAIN-PAPER EXPORT =====================
# These functions create only the individual, LaTeX-assembled main-paper panels.
# They reuse the already-computed ensemble tables and spatial fields. They do not
# alter model state, scenario definitions, metrics or stochastic seeds.

FINAL_PAPER_DPI = 320
FINAL_MAP_CMAP = "turbo"


def _final_paper_path(filename):
    """Return a path inside the clean main-paper-only folder."""
    return FINAL_PAPER_MAIN_DIR / str(filename)


def _final_paper_save(fig, filename):
    path = _final_paper_path(filename)
    fig.savefig(path, dpi=FINAL_PAPER_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Final main plots] saved → {path}")
    return path


def _final_scenario_label(scenario):
    return FINAL_PAPER_SCENARIO_LABELS.get(scenario, str(scenario))


def _final_matrix_mean_sd(df, row_col, value_col, row_order=None):
    """Create a final-paper mean/SD matrix without changing the source table."""
    if df is None or df.empty or row_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    agg = d.groupby([row_col, "scenario"], dropna=False)[value_col].agg(["mean", "std"]).reset_index()
    mean = agg.pivot_table(index=row_col, columns="scenario", values="mean", aggfunc="first")
    sd = agg.pivot_table(index=row_col, columns="scenario", values="std", aggfunc="first")
    if row_order is not None:
        mean = mean.reindex(row_order)
        sd = sd.reindex(mean.index)
    mean = mean.reindex(columns=COOCC_ORDER)
    sd = sd.reindex(index=mean.index, columns=mean.columns).fillna(0.0)
    mean.columns = [_final_scenario_label(s) for s in COOCC_ORDER]
    sd.columns = mean.columns
    return mean, sd


def _final_heatmap(ax, mean, sd, cbar_label, cmap="viridis", fmt=".1f"):
    """Paper heatmap: one mean ± SD annotation per cell and no panel title/footer."""
    if mean is None or mean.empty:
        ax.axis("off")
        return None
    arr = mean.to_numpy(dtype=float)
    sd_arr = sd.reindex(index=mean.index, columns=mean.columns).fillna(0.0).to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    vmin = min(0.0, float(np.nanmin(finite))) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(mean.columns)))
    ax.set_xticklabels(list(mean.columns), rotation=28, ha="right")
    ax.set_yticks(np.arange(len(mean.index)))
    ax.set_yticklabels([str(v) for v in mean.index])
    threshold = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                label = f"{val:{fmt}}\n±{sd_arr[i, j]:{fmt}}"
            else:
                label = "NA"
            ax.text(
                j, i, label, ha="center", va="center", fontsize=7.2,
                color="white" if np.isfinite(val) and val > threshold else "black",
            )
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    return im


def _final_draw_layout_map(ax, layout):
    """Clean layout panel: schools, markets, taps and toilets only; no title/legend."""
    if not isinstance(layout, dict):
        ax.axis("off")
        return
    zones = layout.get("fixed_analysis_zones", None)
    if zones is None or getattr(zones, "empty", True):
        zones = layout.get("outer_cells_active", None)
    styles = [
        ("school", "^", "purple", 22),
        ("market", "s", "green", 20),
        ("tap", "o", "deepskyblue", 24),
        ("toilet", "x", "black", 20),
    ]
    for feature_type, marker, colour, size in styles:
        g = _active_layer(layout, feature_type)
        if g is not None and not g.empty:
            ax.scatter(
                g.geometry.x, g.geometry.y, marker=marker, s=size, c=colour,
                alpha=0.85, linewidths=0.65, zorder=3,
            )
    _draw_city_outline(ax, layout, color="0.35", linewidth=0.65, alpha=0.8, zorder=2)
    if zones is not None and not getattr(zones, "empty", True):
        minx, miny, maxx, maxy = zones.total_bounds
        pad = 0.03 * max(maxx - minx, maxy - miny)
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _save_final_layout_legend():
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(5.4, 0.75))
    ax.axis("off")
    handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="purple", label="Schools", markersize=7),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="green", label="Markets", markersize=7),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="deepskyblue", label="Water taps", markersize=7),
        Line2D([0], [0], marker="x", color="black", label="Public toilets", markersize=7, linestyle="None"),
    ]
    ax.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=8, handletextpad=0.35, columnspacing=1.1)
    _final_paper_save(fig, "fig1_layout_shared_legend.png")


def _save_final_fig1(tables, layout_results):
    layout_files = {
        "baseline": "fig1A_layout_baseline.png",
        "intervD_destinations_only": "fig1B_layout_activity_only.png",
        "intervD_facilities_only": "fig1C_layout_wash_only.png",
        "intervD_area_redistribute": "fig1D_layout_activity_wash.png",
    }
    for scenario in COOCC_ORDER:
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        _final_draw_layout_map(ax, layout_results.get(scenario, {}))
        _final_paper_save(fig, layout_files[scenario])
    _save_final_layout_legend()

    urban = tables.get("urban_form", pd.DataFrame()).copy()
    urban_labels = {
        "hub_area": "Hub area",
        "activity_supply": "Activity supply",
        "wash_supply": "WASH supply",
        "realised_visits": "Realised visits",
    }
    urban_order = ["hub_area", "activity_supply", "wash_supply", "realised_visits"]
    if not urban.empty:
        urban["row_label"] = urban["layer"].map(urban_labels)
        mean, sd = _final_matrix_mean_sd(
            urban, "row_label", "top1_share_pct", [urban_labels[k] for k in urban_order]
        )
        fig, ax = plt.subplots(figsize=(7.7, 4.6))
        _final_heatmap(ax, mean, sd, "Largest-hub share (%)", cmap="YlGnBu", fmt=".1f")
        _final_paper_save(fig, "fig1E_dominant_hub_share.png")

    physical = tables.get("physical", pd.DataFrame()).copy()
    pair_labels = {
        "school_market": "Schools with ≥1 market within 500 m",
        "school_tap": "Schools with ≥1 water tap within 500 m",
        "school_toilet": "Schools with ≥1 public toilet within 500 m",
        "market_tap": "Markets with ≥1 water tap within 500 m",
        "market_toilet": "Markets with ≥1 public toilet within 500 m",
        "home_tap": "Homes with ≥1 water tap within 500 m",
        "home_toilet": "Homes with ≥1 public toilet within 500 m",
    }
    pair_order = [
        "school_market", "school_tap", "school_toilet", "market_tap",
        "market_toilet", "home_tap", "home_toilet",
    ]
    if not physical.empty:
        physical = physical[np.isclose(pd.to_numeric(physical["radius_m"], errors="coerce"), 500.0)].copy()
        physical["pair_code"] = physical["feature_A"].astype(str) + "_" + physical["feature_B"].astype(str)
        physical["row_label"] = physical["pair_code"].map(pair_labels)
        mean, sd = _final_matrix_mean_sd(
            physical, "row_label", "pct_A_with_B_in_radius", [pair_labels[k] for k in pair_order]
        )
        fig, ax = plt.subplots(figsize=(8.4, 6.2))
        _final_heatmap(ax, mean, sd, "Amenities of first-listed type with nearby second-listed amenity (%)", cmap="PuBuGn", fmt=".1f")
        _final_paper_save(fig, "fig1F_physical_cooccurrence_500m.png")


def _save_final_fig2(tables):
    attraction = tables.get("attraction", pd.DataFrame()).copy()
    if attraction.empty:
        return
    attraction["row_label"] = attraction["facility_type"].map(_pretty_feature_label)
    feature_order = [_pretty_feature_label(x) for x in COOCC_FEATURE_ORDER]
    specs = [
        ("visits_per_agent_day", "Completed visits per agent per day", "viridis", "fig2A_visits_per_agent_day.png"),
        ("median_trip_distance_m", "Median completed journey distance (m)", "viridis", "fig2B_completed_journey_distance.png"),
        ("visits_per_facility", "Mean visits per used facility", "viridis", "fig2C_mean_visits_per_used_facility.png"),
    ]
    for metric, cbar, cmap, filename in specs:
        if metric not in attraction.columns:
            continue
        mean, sd = _final_matrix_mean_sd(attraction, "row_label", metric, feature_order)
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        _final_heatmap(ax, mean, sd, cbar, cmap=cmap, fmt=".1f")
        _final_paper_save(fig, filename)


def _save_final_fig3(tables):
    burden = tables.get("pair_burden", pd.DataFrame()).copy()
    if burden.empty:
        return
    pair_order = [
        "school_near_market", "activity_near_tap", "home_near_tap", "market_near_tap", "school_near_tap",
        "activity_near_toilet", "home_near_toilet", "market_near_toilet", "school_near_toilet",
    ]
    burden["row_label"] = burden["pair"].map(lambda x: PAIR_BURDEN_LABELS.get(x, _pretty_pair_label(x)))
    display_order = [PAIR_BURDEN_LABELS.get(k, _pretty_pair_label(k)) for k in pair_order]
    specs = [
        ("mean_visits_per_facility", "Completed visits per selected facility", "PuBuGn", "fig3A_pair_context_visits_per_facility.png", False),
        ("mean_tap_water_exposures_per_facility", "Direct tap-water S→E events per selected tap", "OrRd", "fig3B_pair_context_tap_exposures_per_tap.png", True),
        ("mean_exposures_per_1000_visits", "Direct tap-water S→E events per 1,000 tap visits", "Reds", "fig3C_pair_context_tap_exposures_per_1000_visits.png", True),
    ]
    for metric, cbar, cmap, filename, taps_only in specs:
        d = burden.copy()
        if taps_only:
            d = d[d["target_facility_type"].eq("tap")].copy()
        d = d[d[metric].notna()].copy() if metric in d.columns else pd.DataFrame()
        if d.empty:
            continue
        use_order = [x for x in display_order if x in set(d["row_label"])]
        mean, sd = _final_matrix_mean_sd(d, "row_label", metric, use_order)
        fig, ax = plt.subplots(figsize=(8.8, max(5.1, 0.52 * len(mean.index) + 2.6)))
        _final_heatmap(ax, mean, sd, cbar, cmap=cmap, fmt=".1f")
        _final_paper_save(fig, filename)


def _final_draw_wash_overlay(ax, layout):
    """Peak E+I map overlay: WASH facilities only, no embedded legend or borders."""
    if not isinstance(layout, dict):
        return
    for feature_type, marker, colour, size in [
        ("tap", "o", "deepskyblue", 18),
        ("toilet", "x", "black", 20),
    ]:
        g = _active_layer(layout, feature_type)
        if g is not None and not g.empty:
            ax.scatter(g.geometry.x, g.geometry.y, marker=marker, s=size, c=colour, alpha=0.85, linewidths=0.7, zorder=4)
    _draw_city_outline(ax, layout, color="white", linewidth=0.7, alpha=0.85, zorder=5)


def _save_final_fig4(tables, spatial_acc, layout_results):
    timeseries = tables.get("timeseries", pd.DataFrame()).copy()
    if not timeseries.empty:
        d = timeseries[timeseries["metric"].eq("cumulative_exposure_pct")].copy()
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for scenario in COOCC_ORDER:
            g = d[d["scenario"].eq(scenario)].groupby("tick")["value"].agg(["mean", "std"]).sort_index()
            if g.empty:
                continue
            colour = SCENARIO_COLOURS.get(scenario, None)
            ax.plot(g.index, g["mean"], label=_final_scenario_label(scenario), color=colour, linewidth=1.8)
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=colour, alpha=0.16)
        ax.set_xlabel("Simulation time (h)")
        ax.set_ylabel("Cumulative S→E events (% of agents)")
        ax.legend(frameon=False, fontsize=8)
        ax.margins(x=0)
        fig.tight_layout()
        _final_paper_save(fig, "fig4A_cumulative_SE_events.png")

    outcomes = tables.get("outcomes", pd.DataFrame()).copy()
    if not outcomes.empty:
        metrics = [
            ("peak_6hr_incidence_pct", "Peak 6-hour incidence"),
            ("peak_active_infectious_pct", "Peak active infectious prevalence"),
        ]
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        x = np.arange(len(COOCC_ORDER))
        width = 0.36
        for j, (metric, label) in enumerate(metrics):
            if metric not in outcomes.columns:
                continue
            g = outcomes.groupby("scenario")[metric].agg(["mean", "std"]).reindex(COOCC_ORDER).fillna(0.0)
            offset = (j - 0.5) * width
            ax.bar(x + offset, g["mean"], width=width, yerr=g["std"], capsize=3, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([_final_scenario_label(s) for s in COOCC_ORDER], rotation=22, ha="right")
        ax.set_ylabel("Peak burden (% of agents)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        _final_paper_save(fig, "fig4B_peak_epidemic_burden.png")

    means, vmax = _spatial_mean_field(spatial_acc, "peak_ei")
    if not means:
        return
    xe, ye = spatial_acc["xe"], spatial_acc["ye"]
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    map_files = {
        "baseline": "fig4C_peak_EI_baseline.png",
        "intervD_destinations_only": "fig4D_peak_EI_activity_only.png",
        "intervD_facilities_only": "fig4E_peak_EI_wash_only.png",
        "intervD_area_redistribute": "fig4F_peak_EI_activity_wash.png",
    }
    for scenario in COOCC_ORDER:
        arr = means.get(scenario, np.zeros((len(ye) - 1, len(xe) - 1)))
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.imshow(arr, origin="lower", extent=extent, cmap=FINAL_MAP_CMAP, vmin=0.0, vmax=max(vmax, 1e-9), zorder=1)
        _final_draw_wash_overlay(ax, layout_results.get(scenario, {}))
        ax.axis("off")
        _final_paper_save(fig, map_files[scenario])

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    legend_fig, legend_ax = plt.subplots(figsize=(3.8, 0.65))
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="deepskyblue", label="Water taps", markersize=7),
        Line2D([0], [0], marker="x", color="black", label="Public toilets", markersize=7, linestyle="None"),
    ]
    legend_ax.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=8, columnspacing=1.2)
    _final_paper_save(legend_fig, "fig4_peak_EI_wash_shared_legend.png")

    cbar_fig, cbar_ax = plt.subplots(figsize=(5.0, 0.72))
    cbar_ax.set_visible(False)
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=max(vmax, 1e-9)), cmap=FINAL_MAP_CMAP)
    sm.set_array([])
    cbar = cbar_fig.colorbar(sm, ax=cbar_ax, orientation="horizontal", fraction=0.95, pad=0.1)
    cbar.set_label("Mean E+I density at scenario peak")
    _final_paper_save(cbar_fig, "fig4_peak_EI_shared_colourbar.png")


def _save_final_paper_main_figures(tables, spatial_acc, layout_results):
    """Write only the four final main figures' individual assets to final_paper_main/."""
    print(f"[Final main plots] exporting clean individual panels → {FINAL_PAPER_MAIN_DIR}")
    _paper_style()
    _save_final_fig1(tables, layout_results)
    _save_final_fig2(tables)
    _save_final_fig3(tables)
    _save_final_fig4(tables, spatial_acc, layout_results)
    print("[Final main plots] complete: 21 LaTeX-ready PNG assets were requested.")


COOCCURRENCE_CHECKPOINT_VERSION = 4

def _checkpoint_paths(seed, scenario):
    import gzip
    ckdir=COOCC_OUT/"checkpoints"; ckdir.mkdir(parents=True,exist_ok=True)
    safe=str(scenario).replace("/","_")
    return ckdir/f"v{COOCCURRENCE_CHECKPOINT_VERSION}_seed_{int(seed):03d}_{safe}.pkl.gz", ckdir/f"v{COOCCURRENCE_CHECKPOINT_VERSION}_layout_{safe}.pkl.gz"


def _save_checkpoint(path, payload):
    """Atomically save a checkpoint without sharing a temporary filename across array tasks."""
    import gzip, pickle, uuid
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with gzip.open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _load_checkpoint(path):
    import gzip, pickle
    with gzip.open(path,"rb") as f: return pickle.load(f)

def _run_parameter_screening():
    """Optional structure-only screening; no epidemic simulation and no new final intervention type."""
    if not RUN_PARAMETER_SCREENING:
        return
    rows=[]
    configs=[]
    for k in [1.0,1.5,2.0,2.4]: configs.append(("k_geom",k,{"k_geom":k,"delta_dest":0.8,"delta_fac":0.8}))
    for d in [0.4,0.6,0.8]: configs.append(("delta_dest",d,{"k_geom":2.0,"delta_dest":d,"delta_fac":0.8}))
    for d in [0.4,0.6,0.8]: configs.append(("delta_fac",d,{"k_geom":2.0,"delta_dest":0.8,"delta_fac":d}))
    for pname,pval,over in configs:
        intr=_base_intervention_d_dict("all",f"screen_{pname}_{pval}"); intr.update(over)
        r=run_scenario(ticks=0,seed=RUN_SEED,intervention=intr,save_snapshots=False,log_exposures=False,track_visits=False)
        for rr in _urban_form_rows(r,f"{pname}_{pval}",RUN_SEED): rows.append({"parameter":pname,"value":pval,**rr})
    screen_df=pd.DataFrame(rows)
    screen_df.to_csv(COOCC_OUT/f"{CITY_SLUG}_structural_parameter_screening.csv",index=False)
    if not screen_df.empty:
        fig,axs=plt.subplots(1,3,figsize=(16,4.8))
        for ax,(metric,title) in zip(axs,[("normalised_entropy","Normalised entropy"),("top1_share_pct","Largest-hub share (%)"),("gini","Gini")]):
            for (pname,layer),d in screen_df.groupby(["parameter","layer"]):
                d=d.sort_values("value"); ax.plot(d["value"],d[metric],marker='o',label=f"{pname}: {layer}")
            ax.set_title(title); ax.set_xlabel("Parameter value"); ax.grid(alpha=.25)
        axs[-1].legend(fontsize=6,bbox_to_anchor=(1.02,1),loc='upper left'); fig.suptitle("Structure-only parameter screening (no epidemic simulation)"); fig.tight_layout(); fig.savefig(COOCC_OUT/f"{CITY_SLUG}_structural_parameter_screening.png",dpi=250,bbox_inches='tight'); plt.close(fig)


def run_cooccurrence_ensemble():
    seeds=list(range(COOCCURRENCE_SEED_START,COOCCURRENCE_SEED_START+max(1,COOCCURRENCE_N_SEEDS)))
    print(f"[Co-occurrence ensemble] scenarios={list(COOCC_SCENARIOS)} | seeds={len(seeds)} ({seeds[0]}..{seeds[-1]}) | Sobol={RUN_SOBOL} | resume={COOCCURRENCE_RESUME} | worker_only={COOCCURRENCE_WORKER_ONLY} | aggregate_only={COOCCURRENCE_AGGREGATE_ONLY}")
    if COOCCURRENCE_AGGREGATE_ONLY:
        missing = []
        for seed in seeds:
            for scenario in COOCC_ORDER:
                checkpoint_path, _ = _checkpoint_paths(seed, scenario)
                if not checkpoint_path.exists():
                    missing.append(f"seed={seed}, scenario={scenario}")
        if missing:
            preview = "; ".join(missing[:12])
            extra = "" if len(missing) <= 12 else f"; ... plus {len(missing) - 12} more"
            raise RuntimeError(
                "Aggregate-only mode will not rerun missing seeds. Missing checkpoints: " + preview + extra
            )
        print(f"[Co-occurrence ensemble] aggregate-only mode: loading all {len(seeds)} seed checkpoints without simulation.")
    bag_names=["physical","urban_form","attraction","contact","covisitation","risk","role_risk","agent_risk","tap_level","tap_contamination","pair_burden","tap_typology","activity_facility_cohort","school_wash_alignment","school_exposure_pathway","zone_layers","zone_corr","outcomes","timeseries","legacy_trip","legacy_facility","exposure_source","gravity","paper_summary"]
    bags={k:[] for k in bag_names}
    layout_results={}
    xe,ye=_spatial_grid_setup(); spatial={"xe":xe,"ye":ye,"exposure":{s:[] for s in COOCC_ORDER},"initial_ei":{s:[] for s in COOCC_ORDER},"peak_ei":{s:[] for s in COOCC_ORDER},"final_ei":{s:[] for s in COOCC_ORDER},"od":{s:[] for s in COOCC_ORDER}}

    # Load one saved fixed layout per scenario when resuming.
    if COOCCURRENCE_RESUME:
        for scenario in COOCC_ORDER:
            _,lp=_checkpoint_paths(seeds[0],scenario)
            if lp.exists():
                try: layout_results[scenario]=_load_checkpoint(lp)
                except Exception as e: print(f"[Resume] could not load layout {scenario}: {e}")

    for seed in tqdm(seeds,desc=f"{len(seeds)}-seed co-occurrence ensemble",leave=True):
        for scenario,intr0 in COOCC_SCENARIOS.items():
            cp,lp=_checkpoint_paths(seed,scenario)
            if (COOCCURRENCE_RESUME or COOCCURRENCE_AGGREGATE_ONLY) and cp.exists():
                try:
                    payload=_load_checkpoint(cp)
                    for k in bag_names: bags[k].extend(payload.get("bags",{}).get(k,[]))
                    spatial["exposure"][scenario].append(payload["spatial"]["exposure"])
                    spatial["initial_ei"][scenario].append(payload["spatial"]["initial_ei"])
                    spatial["peak_ei"][scenario].append(payload["spatial"]["peak_ei"])
                    spatial["final_ei"][scenario].append(payload["spatial"]["final_ei"])
                    spatial["od"][scenario].append(payload["spatial"]["od"])
                    if scenario not in layout_results and payload.get("layout") is not None: layout_results[scenario]=payload["layout"]
                    continue
                except Exception as e:
                    print(f"[Resume] checkpoint failed for seed={seed} scenario={scenario}; rerunning: {e}")

            intr=None if intr0 is None else dict(intr0)
            if intr is not None:
                intr["seed"]=RUN_SEED; intr["layout_seed"]=RUN_SEED
            res=run_scenario(ticks=RUN_TICKS,seed=seed,intervention=intr,save_snapshots=False,log_exposures=True,track_visits=True,show_progress=False)
            layout={
                "initial_agents": res.get("initial_agents", pd.DataFrame()).copy(),
                "generic_active": res.get("generic_active"), "schools_active": res.get("schools_active"),
                "markets_active": res.get("markets_active"), "taps_active": res.get("taps_active"),
                "toilets_active": res.get("toilets_active"), "inner_hubs_active": res.get("inner_hubs_active"),
                "outer_cells_active": res.get("outer_cells_active"),
                "fixed_analysis_zones": res.get("fixed_analysis_zones"),
            }
            if scenario not in layout_results:
                layout_results[scenario]=layout
                try: _save_checkpoint(lp,layout)
                except Exception as e: print(f"[Checkpoint] layout save failed {scenario}: {e}")

            _seed_dir = COOCC_OUT / "per_seed_events" / f"seed_{seed:03d}"
            _seed_dir.mkdir(parents=True, exist_ok=True)
            res.get("completed_visits", pd.DataFrame()).drop(columns=["facility_wkb"], errors="ignore").to_csv(_seed_dir / f"{scenario}_completed_visits.csv.gz", index=False, compression="gzip")
            res.get("exposures", pd.DataFrame()).drop(columns=["exposure_facility_wkb"], errors="ignore").to_csv(_seed_dir / f"{scenario}_exposure_events.csv.gz", index=False, compression="gzip")
            res.get("initial_agents", pd.DataFrame()).to_csv(_seed_dir / f"{scenario}_initial_agents.csv.gz", index=False, compression="gzip")
            res.get("tap_contamination_records", pd.DataFrame()).drop(columns=[], errors="ignore").to_csv(_seed_dir / f"{scenario}_tap_contamination_provenance.csv.gz", index=False, compression="gzip")

            local={k:[] for k in bag_names}
            local["physical"]=_nearest_pair_rows(res,scenario,seed)
            local["urban_form"]=_urban_form_rows(res,scenario,seed)
            local["attraction"]=_amenity_attraction_rows(res,scenario,seed)
            local["contact"]=_contact_opportunity_rows(res,scenario,seed)
            local["covisitation"]=_covisitation_rows(res,scenario,seed)
            rr,role_rr,agent_rr=_agent_risk_rows(res,scenario,seed)
            local["risk"]=rr; local["role_risk"]=role_rr; local["agent_risk"]=agent_rr.to_dict("records")
            local["tap_level"]=_tap_level_rows(res,scenario,seed)
            local["tap_contamination"]=_tap_contamination_rows(res,scenario,seed)
            local["pair_burden"]=_feature_pair_burden_rows(res,scenario,seed)
            local["tap_typology"]=_tap_typology_rows(res,scenario,seed)
            local["activity_facility_cohort"]=_activity_facility_cohort_rows(res,scenario,seed)
            local["school_wash_alignment"]=_school_wash_alignment_rows(res,scenario,seed)
            local["school_exposure_pathway"]=_school_exposure_pathway_rows(res,scenario,seed)
            zr,zc=_zone_layer_rows(res,scenario,seed); local["zone_layers"]=zr; local["zone_corr"]=zc
            triprow,fbrow,srcrows,gravrow=_legacy_ensemble_rows(res,scenario,seed,zr)
            local["legacy_trip"]=[triprow]; local["legacy_facility"]=[fbrow]; local["exposure_source"]=srcrows; local["gravity"]=[gravrow]
            oo,tt=_core_outcome_and_timeseries(res,scenario,seed); local["outcomes"]=[oo]; local["timeseries"]=tt
            local["paper_summary"]=_paper_summary_rows(res,scenario,seed,local)
            for k in bag_names: bags[k].extend(local[k])

            exp=res.get("exposures",pd.DataFrame())
            ecoords=exp[["exposure_x","exposure_y"]].dropna().to_numpy(dtype=float) if exp is not None and not exp.empty else np.empty((0,2))
            exp_field=_density_field(ecoords,xe,ye); spatial["exposure"][scenario].append(exp_field)
            _init=res.get("initial_agents",pd.DataFrame())
            if _init is not None and not _init.empty:
                _ip=_init[_init["initial_state"].isin(["E","I"])].dropna(subset=["home_x","home_y"])
                icoords=_ip[["home_x","home_y"]].to_numpy(dtype=float) if len(_ip) else np.empty((0,2))
            else: icoords=np.empty((0,2))
            initial_field=_density_field(icoords,xe,ye); spatial["initial_ei"][scenario].append(initial_field)
            peak=res.get("peak_snapshot",None)
            if peak is not None and not getattr(peak,"empty",True):
                pp=peak[peak["state"].isin(["E","I"])]
                pcoords=np.column_stack((pp.geometry.x,pp.geometry.y)) if len(pp) else np.empty((0,2))
            else: pcoords=np.empty((0,2))
            peak_field=_density_field(pcoords,xe,ye); spatial["peak_ei"][scenario].append(peak_field)
            final=res.get("final_snapshot",None)
            if final is not None and not getattr(final,"empty",True):
                fp=final[final["state"].isin(["E","I"])]
                fgeom=fp.geometry if hasattr(fp,"geometry") else fp.get("current_location",pd.Series(dtype=object))
                fcoords=np.column_stack((fgeom.x,fgeom.y)) if len(fp) else np.empty((0,2))
            else: fcoords=np.empty((0,2))
            final_field=_density_field(fcoords,xe,ye); spatial["final_ei"][scenario].append(final_field)
            od_field=_fixed_zone_od_matrix(res); spatial["od"][scenario].append(od_field)
            try:
                _save_checkpoint(cp,{"bags":local,"spatial":{"exposure":exp_field,"initial_ei":initial_field,"peak_ei":peak_field,"final_ei":final_field,"od":od_field},"layout":layout if seed==seeds[0] else None})
            except Exception as e: print(f"[Checkpoint] seed={seed} scenario={scenario} save failed: {e}")

    tables={k:pd.DataFrame(v) for k,v in bags.items()}
    if COOCCURRENCE_WORKER_ONLY:
        print(f"[Co-occurrence worker] saved checkpoint/event outputs for seeds {seeds[0]}..{seeds[-1]}; shared summaries and figures intentionally skipped.")
        return tables,layout_results

    # The aggregate-only job (or a normal serial run) is the only writer for
    # all-seed tables and figures. This guarantees that every mean and SD uses
    # the complete requested ensemble rather than one array task's seed block.
    for k,df in tables.items():
        df.to_csv(COOCC_OUT/f"{CITY_SLUG}_{k}_per_seed.csv",index=False)
        if not df.empty and "seed" in df.columns:
            group_cols = [c for c in df.columns if c != "seed" and not pd.api.types.is_numeric_dtype(df[c])]
            numeric_cols = [c for c in df.columns if c != "seed" and pd.api.types.is_numeric_dtype(df[c])]
            if group_cols and numeric_cols:
                try:
                    summary = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean","std"]).reset_index()
                    summary.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in summary.columns]
                    summary.to_csv(COOCC_OUT/f"{CITY_SLUG}_{k}_mean_sd.csv",index=False)
                except Exception as e: print(f"[Co-occurrence summary] {k} skipped: {e}")
    if RUN_DIAGNOSTIC_PLOTS:
        _save_ensemble_plots(tables,spatial,layout_results)
        print("[Plots 1/2] Core ensemble figures complete.")
        _save_additional_ensemble_plots(tables,spatial,layout_results)
        print("[Plots 2/2] Additional diagnostic/paper-ready figures complete.")
    else:
        print("[Diagnostic plots] skipped because RUN_DIAGNOSTIC_PLOTS=False.")
    if RUN_PAPER_PLOTS:
        _save_paper_figures(tables,spatial,layout_results)
        _save_final_paper_main_figures(tables,spatial,layout_results)
    else:
        print("[Paper plots] skipped because RUN_PAPER_PLOTS=False.")
    print(f"[Co-occurrence ensemble] complete → {COOCC_OUT}")
    return tables,layout_results



# ===================== CO-OCCURRENCE VALIDATION =====================
# Cohort analyses apply time-window definitions independently of
# the core model mechanisms.
# HPC dependency loading is handled by the environment.

# Default to clean maps.  Set SHOW_INTERNAL_MAP_BORDERS=1 only for a debugging run.
SHOW_INTERNAL_MAP_BORDERS = bool(int(os.environ.get("SHOW_INTERNAL_MAP_BORDERS", "0")))

# These are the actual string labels emitted by _agent_risk_rows().  Keeping this
# mapping here avoids treating a temporal sequence as if it were a distance measure.
COHORT_LABELS.update({
    "Activity → tap within 6 h": "Activity visit followed by tap visit within 6 h",
    "Activity + WASH, same day": "Activity and WASH visits on the same day",
    "Activity + WASH, different days": "Activity and WASH visits on different days",
    "Activity only": "Activity visits only",
    "WASH only": "WASH visits only",
    "No activity or WASH visits": "No recorded activity or WASH visits",
    "School + tap on same day": "School and tap visits on the same day",
    "School without same-day tap": "School visits without a same-day tap visit",
    "Market + tap on same day": "Market and tap visits on the same day",
    "Market without same-day tap": "Market visits without a same-day tap visit",
    "Activity + tap on same day": "Activity and tap visits on the same day",
    "Activity without same-day tap": "Activity visits without a same-day tap visit",
})


def _day_end_tick(horizon_day):
    """Last included tick for a day-specific outcome horizon."""
    return min(int(RUN_TICKS) - 1, int(horizon_day * TICKS_PER_DAY) - 1)


def _visits_before_horizon_and_first_exposure(visits, first_exposure, horizon_day):
    """Keep only visits that had occurred by an outcome horizon and before first S→E.

    This prevents both forms of leakage:
    1. post-exposure leakage (a later visit cannot assign an already-exposed agent
       to an activity cohort); and
    2. future-horizon leakage (a day-4 visit cannot appear in a day-2 cohort).
    """
    if visits is None or visits.empty:
        return visits.copy() if isinstance(visits, pd.DataFrame) else pd.DataFrame()
    out = visits.copy()
    out["tick"] = pd.to_numeric(out["tick"], errors="coerce")
    out = out.dropna(subset=["tick"]).copy()
    out["tick"] = out["tick"].astype(int)
    end_tick = _day_end_tick(horizon_day)
    if first_exposure is None or first_exposure.empty:
        return out.loc[out["tick"] <= end_tick].copy()
    cutoff = first_exposure.set_index("agent")["exposure_tick"]
    out["_first_exposure_tick"] = out["agent"].map(cutoff)
    keep = (
        (out["tick"] <= end_tick)
        & (out["_first_exposure_tick"].isna() | (out["tick"] <= out["_first_exposure_tick"]))
    )
    return out.loc[keep].drop(columns=["_first_exposure_tick"]).copy()


def _activity_facility_cohort_rows(result, scenario, seed, radius_m=500.0):
    """Role-specific activity cohorts with horizon-specific, pre-exposure membership.

    The previous cohort table correctly removed visits after first exposure, but one
    membership set was reused for both day-2 and day-3 outcomes.  This replacement
    constructs a separate day-2 cohort and day-3 cohort.  A visit after day 2 is
    therefore never allowed into the denominator of the day-2 result.
    """
    init = _normalise_initial_agents(result)
    if init.empty:
        return []
    first = _first_exposure_events(result)
    visits = _normalise_completed_visits(result, init)
    base = init.merge(first[["agent", "exposure_tick", "exposure_day"]], on="agent", how="left")
    base["exposure_tick"] = pd.to_numeric(base["exposure_tick"], errors="coerce")
    day2_end = _day_end_tick(2)
    day3_end = _day_end_tick(3)
    base["exposed_by_day2"] = base["exposure_tick"].le(day2_end)
    base["exposed_by_day3"] = base["exposure_tick"].le(day3_end)
    base["time_to_exposure_days"] = base["exposure_tick"] / float(TICKS_PER_DAY)

    visits_day2 = _visits_before_horizon_and_first_exposure(visits, first, 2)
    visits_day3 = _visits_before_horizon_and_first_exposure(visits, first, 3)

    taps_c = _coords_from_gdf(_active_layer(result, "tap"))
    toilets_c = _coords_from_gdf(_active_layer(result, "toilet"))
    trees = {
        "tap": cKDTree(taps_c) if len(taps_c) else None,
        "toilet": cKDTree(toilets_c) if len(toilets_c) else None,
    }
    target_role = {"school": "student", "market": "trader"}

    def _cohort_from_window(vwin, facility_type, facility_ids, role):
        if vwin is None or vwin.empty or not facility_ids:
            vv = pd.DataFrame(columns=getattr(vwin, "columns", []))
            visitors = set()
        else:
            vv = vwin[
                vwin["facility_type"].eq(facility_type)
                & vwin["facility_id"].isin(facility_ids)
                & vwin["role"].eq(role)
            ].copy()
            visitors = set(pd.to_numeric(vv.get("agent", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
        cohort = base[
            base["agent"].isin(visitors)
            & base["initial_state"].eq("S")
            & base["role"].eq(role)
        ].copy()
        return vv, visitors, cohort

    rows = []
    for ft in ["school", "market"]:
        facilities = _active_layer(result, ft).copy().reset_index(drop=True)
        if facilities.empty:
            continue
        if "feature_id" not in facilities.columns:
            facilities["feature_id"] = [getattr(geom, "wkb", i) for i, geom in enumerate(facilities.geometry)]
        role = target_role[ft]
        facility_coords = _coords_from_gdf(facilities)
        for prox_type in ["tap", "toilet"]:
            tree = trees[prox_type]
            if tree is None or len(facility_coords) == 0:
                has_nearby = np.zeros(len(facilities), dtype=bool)
            else:
                has_nearby = np.asarray(
                    [len(x) > 0 for x in tree.query_ball_point(facility_coords, r=radius_m)],
                    dtype=bool,
                )
            for prox_label, mask in [
                (f"{prox_type} within {int(radius_m)} m", has_nearby),
                (f"no {prox_type} within {int(radius_m)} m", ~has_nearby),
            ]:
                fids = set(facilities.loc[mask, "feature_id"].dropna().tolist())
                v2, visitors2, cohort2 = _cohort_from_window(visits_day2, ft, fids, role)
                v3, visitors3, cohort3 = _cohort_from_window(visits_day3, ft, fids, role)
                early3 = cohort3[cohort3["exposure_tick"].le(day3_end)].copy()
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "facility_type": ft,
                    "role_target": role,
                    "proximity_type": prox_type,
                    "proximity_class": prox_label,
                    "radius_m": float(radius_m),
                    "cohort_definition": "role-specific visits observed by each outcome horizon and no later than first exposure",
                    "n_facilities": int(mask.sum()),
                    # Horizon-specific denominators: use these beside the relevant outcome.
                    "n_pre_exposure_role_visitors_day2": int(len(visitors2)),
                    "n_initially_susceptible_visitors_day2": int(len(cohort2)),
                    "n_pre_exposure_role_visitors_day3": int(len(visitors3)),
                    "n_initially_susceptible_visitors_day3": int(len(cohort3)),
                    # Backward-compatible aliases refer to the day-3 membership window.
                    "n_pre_exposure_role_visitors": int(len(visitors3)),
                    "n_initially_susceptible_visitors": int(len(cohort3)),
                    "exposed_by_day2_pct": 100.0 * float(cohort2["exposed_by_day2"].mean()) if len(cohort2) else np.nan,
                    "exposed_by_day3_pct": 100.0 * float(cohort3["exposed_by_day3"].mean()) if len(cohort3) else np.nan,
                    # This is deliberately restricted to exposures occurring by day 3.
                    "median_time_to_exposure_days": float(early3["time_to_exposure_days"].median()) if not early3.empty else np.nan,
                    "mean_visits_per_facility_day2": float(len(v2) / max(int(mask.sum()), 1)) if int(mask.sum()) else np.nan,
                    "mean_visits_per_facility_day3": float(len(v3) / max(int(mask.sum()), 1)) if int(mask.sum()) else np.nan,
                    "mean_visits_per_facility": float(len(v3) / max(int(mask.sum()), 1)) if int(mask.sum()) else np.nan,
                })
    return rows


def _heatmap_mean_sd_n_ax(ax, mean, sd, nmean=None, title="", cbar_label="", cmap="viridis", fmt=".1f", n_prefix="n≈"):
    """Mean ± SD heatmap with a third line showing the average cohort denominator."""
    im = _heatmap_mean_sd_ax(ax, mean, sd, title, cbar_label, cmap=cmap, fmt=fmt)
    if im is None or nmean is None or getattr(nmean, "empty", True):
        return im
    n_arr = nmean.reindex(index=mean.index, columns=mean.columns).to_numpy(dtype=float)
    arr = mean.to_numpy(dtype=float)
    threshold = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            nval = n_arr[i, j]
            if np.isfinite(nval):
                ax.text(
                    j, i + 0.31, f"{n_prefix}{nval:.0f}", ha="center", va="center", fontsize=5.8,
                    color="white" if np.isfinite(arr[i, j]) and arr[i, j] > threshold else "black",
                )
    return im


def _scenario_feature_heatmap_data_with_n(df, feature_col, value_col, n_col, feature_order=None):
    mean, sd, agg = _scenario_feature_heatmap_data(df, feature_col, value_col, feature_order)
    if df is None or df.empty or n_col not in df.columns:
        return mean, sd, pd.DataFrame(), agg
    ndf = df.copy()
    ndf[n_col] = pd.to_numeric(ndf[n_col], errors="coerce")
    n_agg = ndf.groupby([feature_col, "scenario"], dropna=False)[n_col].mean().reset_index(name="n_mean")
    nmean = n_agg.pivot_table(index=feature_col, columns="scenario", values="n_mean", aggfunc="first")
    if feature_order is not None:
        nmean = nmean.reindex(index=feature_order)
    nmean = nmean.reindex(columns=COOCC_ORDER)
    nmean.columns = [COOCC_LABELS.get(c, str(c)) for c in nmean.columns]
    return mean, sd, nmean, agg


def _save_individual_heatmap_panel(stem, mean, sd, title, cbar_label, cmap, fmt, section="supplement", nmean=None, note=None):
    if mean is None or mean.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.52 * len(mean.index) + 2.5)))
    if nmean is None:
        _heatmap_mean_sd_ax(ax, mean, sd, title, cbar_label, cmap=cmap, fmt=fmt)
    else:
        _heatmap_mean_sd_n_ax(ax, mean, sd, nmean, title, cbar_label, cmap=cmap, fmt=fmt)
    footer = note if note else _seed_note()
    ax.text(0.01, -0.23, footer, transform=ax.transAxes, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(_paper_path_refined(stem, section), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _draw_spatial_overlay(ax, layout, overlay="wash", show_legend=False):
    """Draw clean context overlays; generic activity is a faint density layer, not thousands of dots."""
    if not isinstance(layout, dict):
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    if overlay == "all_pois":
        generic = _active_layer(layout, "generic_activity")
        if not generic.empty:
            try:
                ax.hexbin(
                    generic.geometry.x, generic.geometry.y, gridsize=32, mincnt=1,
                    cmap="Oranges", alpha=0.16, linewidths=0, zorder=2,
                )
            except Exception:
                ax.scatter(generic.geometry.x, generic.geometry.y, s=2, c="orange", alpha=0.08, linewidths=0, zorder=2)
            handles.append(Patch(facecolor="orange", alpha=0.35, label="Generic activity density"))
        for ft, mk, co, size, label in [
            ("school", "^", "purple", 18, "Schools"),
            ("market", "s", "green", 16, "Markets"),
        ]:
            g = _active_layer(layout, ft)
            if not g.empty:
                ax.scatter(g.geometry.x, g.geometry.y, marker=mk, s=size, c=co, alpha=0.75, linewidths=0.4, zorder=4)
                handles.append(Line2D([0], [0], marker=mk, color="w", markerfacecolor=co, label=label, markersize=6))
    for ft, mk, co, size, label in [
        ("tap", "o", "deepskyblue", 18, "Water taps"),
        ("toilet", "x", "black", 20, "Toilets"),
    ]:
        g = _active_layer(layout, ft)
        if not g.empty:
            ax.scatter(g.geometry.x, g.geometry.y, marker=mk, s=size, c=co, alpha=0.85, linewidths=0.7, zorder=5)
            handles.append(Line2D([0], [0], marker=mk, color="black" if mk == "x" else "w", markerfacecolor=co if mk != "x" else "none", label=label, markersize=6, linestyle="None"))
    _draw_city_outline(ax, layout, color="white", linewidth=0.8, alpha=0.9, zorder=6)
    if SHOW_INTERNAL_MAP_BORDERS:
        # Debug-only: kept off by default so supervisor-facing maps stay readable.
        zones = layout.get("fixed_analysis_zones", layout.get("outer_cells_active", None))
        hubs = layout.get("inner_hubs_active", None)
        if zones is not None and not getattr(zones, "empty", True):
            zones.boundary.plot(ax=ax, color="0.70", linewidth=0.35, alpha=0.45, zorder=6)
        if hubs is not None and not getattr(hubs, "empty", True):
            hubs.boundary.plot(ax=ax, color="0.35", linewidth=0.45, alpha=0.45, zorder=7)
    if show_legend and handles:
        # Remove repeated labels while preserving the first occurrence.
        dedup = {}
        for h in handles:
            label = h.get_label()
            if label and label not in dedup:
                dedup[label] = h
        ax.legend(dedup.values(), dedup.keys(), loc="lower left", fontsize=6.4, frameon=True, handlelength=1.2)


def _plot_paper_spatial_density_map(spatial_acc, layout_results, field_key="exposure", overlay="wash", stem="paper_map", title=""):
    """Save a four-scenario map plus a standalone PNG for each scenario panel."""
    if spatial_acc is None or field_key not in spatial_acc:
        return
    means, vmax = _spatial_mean_field(spatial_acc, field_key)
    if not means:
        return
    xe, ye = spatial_acc["xe"], spatial_acc["ye"]
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    section = "main" if field_key == "exposure" else "supplement"
    fig, axs = plt.subplots(1, len(COOCC_ORDER), figsize=(16, 4.8), sharex=True, sharey=True, squeeze=False)
    im = None
    for ax, sc in zip(axs[0], COOCC_ORDER):
        arr = means.get(sc, np.zeros_like(next(iter(means.values()))))
        im = ax.imshow(arr, origin="lower", extent=extent, cmap="turbo", vmin=0, vmax=max(vmax, 1e-9), zorder=1)
        _draw_spatial_overlay(ax, layout_results.get(sc, {}), overlay=overlay, show_legend=(sc == COOCC_ORDER[-1]))
        ax.set_title(COOCC_LABELS.get(sc, sc), fontsize=9)
        ax.axis("off")

        # Each scenario map is also independently reusable in slides/manuscripts.
        single, sax = plt.subplots(figsize=(5.6, 5.2))
        sax.imshow(arr, origin="lower", extent=extent, cmap="turbo", vmin=0, vmax=max(vmax, 1e-9), zorder=1)
        _draw_spatial_overlay(sax, layout_results.get(sc, {}), overlay=overlay, show_legend=True)
        sax.set_title(f"{COOCC_LABELS.get(sc, sc)}\n{title}", fontsize=9)
        sax.axis("off")
        single.colorbar(sax.images[0], ax=sax, fraction=0.046, pad=0.03, label="Mean density across seeds")
        single.text(0.5, 0.01, _seed_note(), ha="center", fontsize=7.5)
        single.tight_layout(rect=(0, 0.03, 1, 1))
        safe_sc = str(sc).replace("/", "_")
        single.savefig(_paper_path_refined(f"{stem}_{safe_sc}", section), dpi=320, bbox_inches="tight")
        plt.close(single)
    if im is not None:
        fig.colorbar(im, ax=axs[0].tolist(), fraction=0.02, pad=0.02, label="Mean density across seeds")
    fig.suptitle(title)
    fig.text(0.5, 0.01, _seed_note(), ha="center", fontsize=8)
    fig.savefig(_paper_path_refined(stem, section), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_fig1_spatial_structure(tables, spatial_acc, layout_results):
    """Refined Figure 1 saved in paper_figures/main/, with standalone panel files."""
    # A: full spatial layout row + each scenario map separately.
    figA, axesA = plt.subplots(1, len(COOCC_ORDER), figsize=(15.5, 4.5), squeeze=False)
    for ax, sc in zip(axesA[0], COOCC_ORDER):
        if sc in layout_results:
            _draw_paper_layout_map(ax, layout_results[sc], sc, show_legend=(sc == COOCC_ORDER[-1]))
        else:
            ax.axis("off")
        if sc in layout_results:
            single, sax = plt.subplots(figsize=(5.4, 5.0))
            _draw_paper_layout_map(sax, layout_results[sc], sc, show_legend=True)
            single.text(0.5, 0.01, "Fixed spatial layout across stochastic seeds", ha="center", fontsize=7.5)
            single.tight_layout(rect=(0, 0.03, 1, 1))
            single.savefig(_paper_path_refined(f"paper_fig1A_spatial_layout_{sc}", "main"), dpi=320, bbox_inches="tight")
            plt.close(single)
    figA.suptitle("Figure 1A. Scenario spatial systems: activity and WASH placement")
    figA.text(0.5, 0.01, "Homes remain fixed in every scenario; only activity and/or WASH locations are redistributed.", ha="center", fontsize=8)
    figA.savefig(_paper_path_refined("paper_fig1A_spatial_layout_clean", "main"), dpi=320, bbox_inches="tight")
    # Backward-compatible root copy retained.
    figA.savefig(_paper_path("paper_fig1A_spatial_layout_clean"), dpi=320, bbox_inches="tight")
    plt.close(figA)

    uf = tables.get("urban_form", pd.DataFrame()).copy()
    pc = tables.get("physical", pd.DataFrame()).copy()
    fig = plt.figure(figsize=(16.8, 9.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.08, 1.0])
    for i, sc in enumerate(COOCC_ORDER):
        ax = fig.add_subplot(gs[0, i])
        if sc in layout_results:
            _draw_paper_layout_map(ax, layout_results[sc], sc, show_legend=(sc == COOCC_ORDER[-1]))
        else:
            ax.axis("off")
    axB = fig.add_subplot(gs[1, 0:2])
    axC = fig.add_subplot(gs[1, 2:4])
    _panel_label(fig.axes[0], "A")
    if not uf.empty:
        d = uf[uf["layer"].isin(["hub_area", "activity_supply", "wash_supply", "realised_visits"])].copy()
        d["layer_label"] = d["layer"].map({
            "hub_area": "Hub area",
            "activity_supply": "Activity supply",
            "wash_supply": "WASH supply",
            "realised_visits": "Realised visits",
        }).fillna(d["layer"])
        m, s, agg = _mean_sd_table(d, ["layer_label"], "top1_share_pct")
        _save_refined_panel_data("paper_fig1B_top1_share", agg, "main")
        _heatmap_mean_sd_ax(axB, m, s, "Largest-hub share by layer", "Top share (%)", cmap="YlGnBu", fmt=".1f")
        _panel_label(axB, "B")
        _save_individual_heatmap_panel("paper_fig1B_top1_share", m, s, "Largest-hub share by layer", "Top share (%)", "YlGnBu", ".1f", "main")
    else:
        axB.axis("off")
    if not pc.empty:
        d = pc[np.isclose(pd.to_numeric(pc["radius_m"], errors="coerce"), 500.0)].copy()
        d["pair"] = d["feature_A"].map(_pretty_feature_label) + "–" + d["feature_B"].map(_pretty_feature_label)
        m, s, agg = _mean_sd_table(d, ["pair"], "pct_A_with_B_in_radius")
        _save_refined_panel_data("paper_fig1C_physical_cooccurrence_500m", agg, "main")
        _heatmap_mean_sd_ax(axC, m, s, "Physical co-occurrence within 500 m", "% of A with B", cmap="YlOrBr", fmt=".1f")
        _panel_label(axC, "C")
        _save_individual_heatmap_panel("paper_fig1C_physical_cooccurrence_500m", m, s, "Physical co-occurrence within 500 m", "% of A with B", "YlOrBr", ".1f", "main")
    else:
        axC.axis("off")
    fig.suptitle("Figure 1. Urban and environmental co-occurrence structure")
    fig.text(0.5, 0.01, _seed_note() + "; structural layout measures have SD = 0 because the layout is fixed across epidemic seeds.", ha="center", fontsize=7.8)
    fig.savefig(_paper_path_refined("paper_fig1_spatial_structure", "main"), dpi=320, bbox_inches="tight")
    fig.savefig(_paper_path("paper_fig1_spatial_structure"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_role_specific_agent_risk(tables):
    """Plot the named school/market/activity same-day tap cohorts requested for interpretation."""
    rr = tables.get("role_risk", pd.DataFrame()).copy()
    if rr.empty or "analysis_type" not in rr.columns:
        return
    rr = rr[rr["analysis_type"].eq("role_specific")].copy()
    if rr.empty:
        return
    desired = [
        ("student", "School + tap on same day"),
        ("student", "School without same-day tap"),
        ("trader", "Market + tap on same day"),
        ("trader", "Market without same-day tap"),
        ("general", "Activity + tap on same day"),
        ("general", "Activity without same-day tap"),
    ]
    rr["cohort_label"] = rr.apply(
        lambda row: f"{str(row['role']).title()}: {COHORT_LABELS.get(str(row['risk_cohort']), str(row['risk_cohort']))}",
        axis=1,
    )
    order = [f"{role.title()}: {COHORT_LABELS.get(label, label)}" for role, label in desired]
    rr = rr[rr["cohort_label"].isin(order)].copy()
    if rr.empty:
        return
    # Avoid showing very small role-specific groups as substantive cells.
    nmean = rr.groupby("cohort_label")["n_agents"].mean()
    order = [x for x in order if x in set(nmean.index) and nmean.loc[x] >= MIN_COHORT_N_FOR_MAIN_PLOT]
    if not order:
        return
    rr = rr[rr["cohort_label"].isin(order)].copy()
    specs = [
        ("risk_day2_pct", "Exposure by day 2", "% exposed", "Reds", ".1f", "supp_role_risk_A_day2"),
        ("risk_day3_pct", "Exposure by day 3", "% exposed", "Reds", ".1f", "supp_role_risk_B_day3"),
        ("median_time_to_exposure_days", "Median time to exposure", "Days", "viridis", ".2f", "supp_role_risk_C_time_to_exposure"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
    for ax, (metric, title, cbar, cmap, fmt, stem) in zip(axs, specs):
        m, s, nmean_grid, agg = _scenario_feature_heatmap_data_with_n(rr, "cohort_label", metric, "n_agents", order)
        _save_refined_panel_data(stem, agg, "supplement")
        _heatmap_mean_sd_n_ax(ax, m, s, nmean_grid, title, cbar, cmap=cmap, fmt=fmt)
        _save_individual_heatmap_panel(stem, m, s, title, cbar, cmap, fmt, "supplement", nmean_grid, note="Role-specific, initially susceptible cohorts; n≈ is the mean cohort size across seeds. " + _seed_note())
    fig.suptitle("Role-specific agent risk after experienced activity–tap co-occurrence")
    fig.text(0.5, 0.01, "These are temporal same-day visit cohorts, not physical-distance categories. n≈ is the mean cohort size across seeds.", ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("supp_role_specific_agent_risk", "supplement"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_agent_experienced_cooccurrence_risk(tables):
    """Supplementary broad temporal cohorts, role-specific cohorts, and activity-facility cohorts."""
    risk = tables.get("risk", pd.DataFrame()).copy()
    ac = tables.get("activity_facility_cohort", pd.DataFrame()).copy()

    if not risk.empty:
        r = risk.copy()
        r["risk_label"] = r["risk_cohort"].map(lambda x: COHORT_LABELS.get(x, str(x)))
        rmain = r[r["horizon_day"].isin([2, 3])].copy() if "horizon_day" in r.columns else pd.DataFrame()
        if not rmain.empty:
            nmean = rmain.groupby("risk_label")["n_agents"].mean()
            keep = nmean[nmean >= MIN_COHORT_N_FOR_MAIN_PLOT].index.tolist()
            rmain = rmain[rmain["risk_label"].isin(keep)].copy()
            specs = [
                (rmain[rmain["horizon_day"].eq(2)], "risk_day2_pct", "Early exposure by experienced sequence (day 2)", "% exposed", "Reds", ".1f", "supp_agent_risk_A_day2"),
                (rmain[rmain["horizon_day"].eq(3)], "risk_day3_pct", "Early exposure by experienced sequence (day 3)", "% exposed", "Reds", ".1f", "supp_agent_risk_B_day3"),
                (rmain[rmain["horizon_day"].eq(3)], "median_time_to_exposure_days", "Median time to early exposure", "Days", "viridis", ".2f", "supp_agent_risk_C_time_to_exposure"),
            ]
            fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
            for ax, (d, metric, title, cbar, cmap, fmt, stem) in zip(axs, specs):
                if d.empty:
                    ax.axis("off"); continue
                order = list(d["risk_label"].drop_duplicates())
                m, s, ngrid, agg = _scenario_feature_heatmap_data_with_n(d, "risk_label", metric, "n_agents", order)
                _save_refined_panel_data(stem, agg, "supplement")
                _heatmap_mean_sd_n_ax(ax, m, s, ngrid, title, cbar, cmap=cmap, fmt=fmt)
                _save_individual_heatmap_panel(stem, m, s, title, cbar, cmap, fmt, "supplement", ngrid, note="Temporal sequence, not physical distance. n≈ is mean cohort size across seeds. " + _seed_note())
            fig.suptitle("Experienced co-occurrence: temporal sequences, not physical distance")
            fig.text(0.5, 0.01, "A sequence means that the same agent visited an activity destination and then a tap within the stated time window. n≈ is the mean cohort size.", ha="center", fontsize=8)
            fig.savefig(_paper_path_refined("supp_agent_experienced_cooccurrence_risk", "supplement"), dpi=320, bbox_inches="tight")
            plt.close(fig)

    _plot_role_specific_agent_risk(tables)

    if not ac.empty:
        ac = ac.copy()
        ac["cohort_label"] = ac["facility_type"].str.title() + " visitors | " + ac["proximity_class"]
        specs = [
            ("exposed_by_day2_pct", "Visitors exposed by day 2", "% exposed", "Reds", ".1f", "supp_activity_facility_cohort_A_day2", "n_initially_susceptible_visitors_day2"),
            ("exposed_by_day3_pct", "Visitors exposed by day 3", "% exposed", "Reds", ".1f", "supp_activity_facility_cohort_B_day3", "n_initially_susceptible_visitors_day3"),
            ("median_time_to_exposure_days", "Median time to early exposure (by day 3)", "Days", "viridis", ".2f", "supp_activity_facility_cohort_C_time", "n_initially_susceptible_visitors_day3"),
        ]
        fig, axs = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
        for ax, (metric, title, cbar, cmap, fmt, stem, ncol) in zip(axs, specs):
            m, s, ngrid, agg = _scenario_feature_heatmap_data_with_n(ac, "cohort_label", metric, ncol)
            _save_refined_panel_data(stem, agg, "supplement")
            _heatmap_mean_sd_n_ax(ax, m, s, ngrid, title, cbar, cmap=cmap, fmt=fmt)
            _save_individual_heatmap_panel(stem, m, s, title, cbar, cmap, fmt, "supplement", ngrid, note="Membership uses visits observed by the relevant outcome horizon and no later than first exposure. n≈ is mean cohort size. " + _seed_note())
        fig.suptitle("Role-specific activity visitor cohorts by nearby WASH")
        fig.text(0.5, 0.01, "Day-2 and day-3 denominators are constructed separately; later visits cannot enter an earlier outcome cohort. n≈ is the mean cohort size.", ha="center", fontsize=8)
        fig.savefig(_paper_path_refined("supp_activity_facility_cohort", "supplement"), dpi=320, bbox_inches="tight")
        plt.close(fig)


def _plot_school_wash_alignment(tables):
    """School–WASH alignment with standalone PNG for every panel."""
    sw = tables.get("school_wash_alignment", pd.DataFrame()).copy()
    if sw.empty:
        return
    sw = sw[sw["facility_type"].eq("school")].copy()
    if sw.empty:
        return
    summary = sw.groupby(["scenario", "seed"], dropna=False).agg(
        pct_schools_tap_500=("n_taps_within_500m", lambda x: 100.0 * float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean()) if len(x) else np.nan),
        pct_schools_toilet_500=("n_toilets_within_500m", lambda x: 100.0 * float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean()) if len(x) else np.nan),
        median_school_tap_m=("nearest_tap_m", "median"),
        median_school_toilet_m=("nearest_toilet_m", "median"),
    ).reset_index()
    _save_refined_panel_data("supp_school_wash_alignment", summary, "supplement")
    specs = [
        ("pct_schools_tap_500", "% of schools with tap within 500 m", "% of schools", "supp_school_wash_alignment_A_tap_share"),
        ("pct_schools_toilet_500", "% of schools with toilet within 500 m", "% of schools", "supp_school_wash_alignment_B_toilet_share"),
        ("median_school_tap_m", "Median school-to-nearest-tap distance", "Metres", "supp_school_wash_alignment_C_tap_distance"),
        ("median_school_toilet_m", "Median school-to-nearest-toilet distance", "Metres", "supp_school_wash_alignment_D_toilet_distance"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for ax, (metric, title, ylabel, stem) in zip(axs.ravel(), specs):
        _bar_mean_sd_ax(ax, summary, metric, title, ylabel)
        single, sax = plt.subplots(figsize=(7.8, 4.9))
        _bar_mean_sd_ax(sax, summary, metric, title, ylabel)
        sax.text(0.01, -0.22, _seed_note(), transform=sax.transAxes, fontsize=7.5)
        single.tight_layout()
        single.savefig(_paper_path_refined(stem, "supplement"), dpi=320, bbox_inches="tight")
        plt.close(single)
    fig.suptitle("School–WASH alignment diagnostic")
    fig.text(0.5, 0.01, "Structural diagnostic: checks whether D1 moved schools into different proximity to the unchanged WASH layer.", ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("supp_school_wash_alignment", "supplement"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _school_consistency_panel(ax, d, prox, metric, title, ylabel, annotate_n=True):
    classes = [f"{prox} within 500 m", f"no {prox} within 500 m"]
    n_col = "n_initially_susceptible_visitors_day2" if metric == "exposed_by_day2_pct" else "n_initially_susceptible_visitors_day3"
    width = 0.34
    x = np.arange(len(COOCC_ORDER))
    for j, cl in enumerate(classes):
        sub = d[d["proximity_class"].eq(cl)].copy()
        g = sub.groupby("scenario")[metric].agg(["mean", "std"]).reindex(COOCC_ORDER)
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, g["mean"], width, yerr=g["std"].fillna(0), capsize=3, label=cl)
        for k, sc in enumerate(COOCC_ORDER):
            raw = sub[sub["scenario"].eq(sc)][metric].dropna().to_numpy(dtype=float)
            if len(raw):
                jitter = np.linspace(-0.05, 0.05, len(raw)) if len(raw) > 1 else np.array([0.0])
                ax.scatter(np.full(len(raw), x[k] + offset) + jitter, raw, s=18, color="black", alpha=0.72, zorder=4)
            if annotate_n:
                nvals = pd.to_numeric(sub[sub["scenario"].eq(sc)].get(n_col, pd.Series(dtype=float)), errors="coerce").dropna()
                if len(nvals):
                    height = g.loc[sc, "mean"] if sc in g.index else np.nan
                    if np.isfinite(height):
                        spread = g.loc[sc, "std"] if sc in g.index and np.isfinite(g.loc[sc, "std"]) else 0.0
                        ax.text(x[k] + offset, height + spread + max(0.8, 0.02 * max(1.0, abs(height))), f"n≈{nvals.mean():.0f}", ha="center", va="bottom", fontsize=6.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([COOCC_LABELS[s] for s in COOCC_ORDER], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title + f"\n({prox} proximity; raw seed points shown)")
    ax.legend(fontsize=7, frameon=True)


def _plot_school_cohort_seed_consistency(tables):
    """D1 school cohort check with raw seeds, horizon-specific n, and separate PNG panels."""
    ac = tables.get("activity_facility_cohort", pd.DataFrame()).copy()
    if ac.empty:
        return
    ac = ac[ac["facility_type"].eq("school")].copy()
    if ac.empty:
        return
    specs = [
        ("tap", "exposed_by_day2_pct", "Students visiting schools: exposure by day 2", "% exposed", "supp_school_cohort_seed_consistency_A_tap_day2"),
        ("toilet", "exposed_by_day2_pct", "Students visiting schools: exposure by day 2", "% exposed", "supp_school_cohort_seed_consistency_B_toilet_day2"),
        ("tap", "median_time_to_exposure_days", "Students visiting schools: median time to early exposure", "Days", "supp_school_cohort_seed_consistency_C_tap_time"),
        ("toilet", "median_time_to_exposure_days", "Students visiting schools: median time to early exposure", "Days", "supp_school_cohort_seed_consistency_D_toilet_time"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, (prox, metric, title, ylabel, stem) in zip(axs.ravel(), specs):
        d = ac[ac["proximity_type"].eq(prox)].copy()
        _school_consistency_panel(ax, d, prox, metric, title, ylabel, annotate_n=True)
        single, sax = plt.subplots(figsize=(8.8, 5.4))
        _school_consistency_panel(sax, d, prox, metric, title, ylabel, annotate_n=True)
        sax.text(0.01, -0.27, "Each dot is one seed; n≈ is the mean initially susceptible cohort size. " + _seed_note(), transform=sax.transAxes, fontsize=7.3)
        single.tight_layout()
        single.savefig(_paper_path_refined(stem, "supplement"), dpi=320, bbox_inches="tight")
        plt.close(single)
    fig.suptitle("D1 school-cohort consistency diagnostic")
    fig.text(0.5, 0.01, "Each dot is one seed. Day-2 and day-3 membership windows are built separately; later visits cannot enter earlier cohorts.", ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("supp_school_cohort_seed_consistency", "supplement"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_school_exposure_pathway(tables):
    """Student school-to-exposure diagnostics with standalone PNG panels."""
    sp = tables.get("school_exposure_pathway", pd.DataFrame()).copy()
    if sp.empty:
        return
    summary = sp.groupby(["scenario", "seed"], dropna=False).agg(
        pct_tap_water_exposure=("exposure_source", lambda x: 100.0 * float(pd.Series(x).eq("tap_water").mean()) if len(x) else np.nan),
        pct_exposure_tap_near_last_school=("exposure_tap_within_500m_of_last_school", lambda x: 100.0 * float(pd.Series(x).fillna(False).astype(bool).mean()) if len(x) else np.nan),
        median_school_to_exposure_tap_m=("distance_last_school_to_exposure_location_m", "median"),
        median_hours_school_to_exposure=("time_from_school_visit_to_exposure_h", "median"),
        n_student_paths=("agent", "nunique"),
    ).reset_index()
    _save_refined_panel_data("supp_school_exposure_pathway", summary, "supplement")
    specs = [
        ("pct_tap_water_exposure", "% of student pathways with tap-water exposure", "% of pathways", "supp_school_exposure_pathway_A_tap_source"),
        ("pct_exposure_tap_near_last_school", "% of exposure taps within 500 m of last school", "% of pathways", "supp_school_exposure_pathway_B_near_tap"),
        ("median_school_to_exposure_tap_m", "Median distance: last school to exposure tap", "Metres", "supp_school_exposure_pathway_C_distance"),
        ("median_hours_school_to_exposure", "Median time: last school visit to exposure", "Hours", "supp_school_exposure_pathway_D_time"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for ax, (metric, title, ylabel, stem) in zip(axs.ravel(), specs):
        _bar_mean_sd_ax(ax, summary, metric, title, ylabel)
        single, sax = plt.subplots(figsize=(7.8, 4.9))
        _bar_mean_sd_ax(sax, summary, metric, title, ylabel)
        sax.text(0.01, -0.22, _seed_note(), transform=sax.transAxes, fontsize=7.5)
        single.tight_layout()
        single.savefig(_paper_path_refined(stem, "supplement"), dpi=320, bbox_inches="tight")
        plt.close(single)
    fig.suptitle("Student school-to-exposure pathway diagnostic")
    fig.text(0.5, 0.01, "Diagnostic only: sequence/proximity consistency, not proof that schools or toilets directly transmit infection.", ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("supp_school_exposure_pathway", "supplement"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_tap_typology(tables):
    """Supplementary tap typology with a reusable PNG for every panel."""
    tt = tables.get("tap_typology", pd.DataFrame()).copy()
    if tt.empty:
        return
    specs = [
        ("n_taps", "Number of taps", "Count", "paper_tap_typology_A_counts"),
        ("mean_visits_per_tap", "Visits per tap", "Visits / tap", "paper_tap_typology_B_visits"),
        ("mean_tap_water_exposures_per_tap", "Tap-water exposures per tap", "Exposures / tap", "paper_tap_typology_C_exposures"),
        ("mean_exposures_per_1000_visits", "Exposure intensity", "Exposures / 1,000 visits", "paper_tap_typology_D_rate"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (metric, title, cbar, stem) in zip(axs.ravel(), specs):
        m, s, agg = _mean_sd_table(tt, ["tap_typology"], metric)
        _save_refined_panel_data(stem, agg, "supplement")
        _heatmap_mean_sd_ax(ax, m, s, title, cbar, cmap="magma", fmt=".1f")
        _save_individual_heatmap_panel(stem, m, s, title, cbar, "magma", ".1f", "supplement")
    fig.suptitle("Tap typology: residential and activity context")
    fig.text(0.5, 0.01, "Interpret categories cautiously where the count panel shows few taps. " + _seed_note(), ha="center", fontsize=8)
    fig.savefig(_paper_path_refined("paper_tap_typology", "supplement"), dpi=320, bbox_inches="tight")
    plt.close(fig)


# Checkpoint version identifier for horizon-specific cohort outputs.

COOCCURRENCE_CHECKPOINT_VERSION = 5


# ===================== ROLE-SPECIFIC HORIZON COHORTS =====================
# The named school/market same-day tap cohorts are also made horizon-specific.
# This avoids using day-7 membership to describe a day-2 or day-3 outcome.

def _agent_risk_rows(result, scenario, seed):
    """Pre-exposure broad and role-specific risk cohorts with horizon-specific membership.

    The ABM is unchanged.  This function only constructs analysis cohorts from
    visits that had occurred by the relevant outcome horizon and no later than
    first S→E exposure.
    """
    init = _normalise_initial_agents(result)
    if init.empty:
        return [], [], pd.DataFrame()
    raw_init = result.get("initial_agents", pd.DataFrame()).copy()
    if "agent" not in raw_init.columns and "agent_id" in raw_init.columns:
        raw_init = raw_init.rename(columns={"agent_id": "agent"})
    if "agent" not in raw_init.columns:
        raw_init = init.copy()
    raw_init["agent"] = pd.to_numeric(raw_init["agent"], errors="coerce")
    raw_init = raw_init.dropna(subset=["agent"]).copy()
    raw_init["agent"] = raw_init["agent"].astype(int)
    for col, default in [("household_id", np.nan), ("home_x", np.nan), ("home_y", np.nan)]:
        if col not in raw_init.columns:
            raw_init[col] = default

    first = _first_exposure_events(result)
    visits_all = _normalise_completed_visits(result, init)
    trips_all = result.get("actual_trips", pd.DataFrame()).copy()

    base = raw_init.merge(
        first[["agent", "exposure_tick", "exposure_day", "exposure_source"]], on="agent", how="left"
    )
    base["became_exposed"] = base["exposure_tick"].notna()
    base["exposure_tick"] = pd.to_numeric(base["exposure_tick"], errors="coerce")
    for h in (1, 2, 3, 7):
        base[f"exposed_by_day{h}"] = base["exposure_tick"].le(_day_end_tick(h))
    base["time_to_exposure_ticks"] = base["exposure_tick"]
    base["time_to_exposure_days"] = base["exposure_tick"] / float(TICKS_PER_DAY)
    exp_tick_map = base.set_index("agent")["exposure_tick"]

    def _window_trips(horizon_day):
        if trips_all is None or trips_all.empty or "tick" not in trips_all.columns:
            return pd.DataFrame(columns=getattr(trips_all, "columns", []))
        d = trips_all.copy()
        d["agent"] = pd.to_numeric(d.get("agent", np.nan), errors="coerce")
        d["tick"] = pd.to_numeric(d["tick"], errors="coerce")
        d = d.dropna(subset=["agent", "tick"]).copy()
        d["agent"] = d["agent"].astype(int)
        d["tick"] = d["tick"].astype(int)
        d["_exp_tick"] = d["agent"].map(exp_tick_map)
        end = _day_end_tick(horizon_day)
        keep = (d["tick"] <= end) & (d["_exp_tick"].isna() | (d["tick"] <= d["_exp_tick"]))
        return d.loc[keep].drop(columns=["_exp_tick"]).copy()

    def _assign_cohort(agent_flags):
        act = agent_flags[["visited_school", "visited_market", "visited_generic_activity"]].any(axis=1)
        wash = agent_flags[["visited_tap", "visited_toilet"]].any(axis=1)
        cohort = pd.Series("No activity or WASH visits", index=agent_flags.index, dtype=object)
        cohort.loc[act & ~wash] = "Activity only"
        cohort.loc[wash & ~act] = "WASH only"
        cohort.loc[act & wash] = "Activity + WASH, different days"
        cohort.loc[agent_flags["activity_wash_same_day"]] = "Activity + WASH, same day"
        cohort.loc[agent_flags["activity_to_tap_within_6h"]] = "Activity → tap within 6 h"
        return cohort

    horizon_tables = {}
    for horizon in (1, 2, 3, 7):
        vwin = _visits_before_horizon_and_first_exposure(visits_all, first, horizon)
        awin = _agent_covisitation_table(result, visits_override=vwin)
        keepcols = [
            "agent", "household_id", "home_x", "home_y", "role", "initial_state",
            "exposure_tick", "exposure_day", "exposure_source", "became_exposed",
            "exposed_by_day1", "exposed_by_day2", "exposed_by_day3", "exposed_by_day7",
            "time_to_exposure_ticks", "time_to_exposure_days",
        ]
        base_for_merge = base.copy()
        for col in keepcols:
            if col not in base_for_merge.columns:
                base_for_merge[col] = np.nan
        # _agent_covisitation_table already includes role/initial_state; use the
        # base versions only to fill missing values and avoid duplicate columns.
        awin = awin.merge(base_for_merge.drop(columns=[c for c in ["role", "initial_state"] if c in base_for_merge.columns])[ [c for c in keepcols if c not in {"role", "initial_state"}] ], on="agent", how="left")
        if "role_x" in awin.columns:
            awin["role"] = awin["role_x"].fillna(awin.get("role_y"))
            awin = awin.drop(columns=[c for c in ["role_x", "role_y"] if c in awin.columns])
        if "initial_state_x" in awin.columns:
            awin["initial_state"] = awin["initial_state_x"].fillna(awin.get("initial_state_y"))
            awin = awin.drop(columns=[c for c in ["initial_state_x", "initial_state_y"] if c in awin.columns])
        t_win = _window_trips(horizon)
        if not t_win.empty and "actual_trip_distance_m" in t_win.columns:
            tg = t_win.groupby("agent").agg(
                total_trip_steps=("tick", "count"),
                total_distance_m=("actual_trip_distance_m", "sum"),
            ).reset_index()
            awin = awin.merge(tg, on="agent", how="left")
        for col in ["total_trip_steps", "total_distance_m"]:
            if col not in awin.columns:
                awin[col] = 0.0
            awin[col] = pd.to_numeric(awin[col], errors="coerce").fillna(0.0)
        awin["risk_cohort"] = _assign_cohort(awin)
        horizon_tables[horizon] = awin

    broad_rows = []
    for horizon, awin in horizon_tables.items():
        susceptible = awin[awin["initial_state"].eq("S")].copy()
        outcome_col = f"exposed_by_day{horizon}"
        for cohort, d in susceptible.groupby("risk_cohort"):
            row = {
                "scenario": scenario, "seed": seed, "risk_cohort": cohort,
                "horizon_day": horizon, "membership_horizon": f"day_{horizon}", "n_agents": len(d),
                "risk_day1_pct": np.nan, "risk_day2_pct": np.nan,
                "risk_day3_pct": np.nan, "risk_day7_pct": np.nan,
                "mean_exposure_tick": float(d.loc[d[outcome_col], "exposure_tick"].mean()) if d[outcome_col].any() else np.nan,
                "median_time_to_exposure_days": float(d.loc[d[outcome_col], "time_to_exposure_days"].median()) if d[outcome_col].any() else np.nan,
                "mean_trip_steps": float(d["total_trip_steps"].mean()),
                "mean_distance_km": float(d["total_distance_m"].mean() / 1000.0),
            }
            row[f"risk_day{horizon}_pct"] = 100.0 * float(d[outcome_col].mean())
            broad_rows.append(row)

    # Preserve an agent-level day-7 table for the existing travel-risk diagnostics.
    day7 = horizon_tables[7].copy()
    day7["scenario"] = scenario
    day7["seed"] = seed
    day7["risk_cohort_day7"] = day7["risk_cohort"]
    day3_lookup = horizon_tables[3].set_index("agent")["risk_cohort"]
    day7["risk_cohort_day3"] = day7["agent"].map(day3_lookup)

    role_rows = []
    # Broad role table: retain the existing day-7 summary for old diagnostic plots.
    susceptible7 = day7[day7["initial_state"].eq("S")].copy()
    for (role, cohort), d in susceptible7.groupby(["role", "risk_cohort_day7"]):
        role_rows.append({
            "scenario": scenario, "seed": seed, "role": role, "risk_cohort": cohort,
            "analysis_type": "broad_cohort", "horizon_day": 7, "membership_horizon": "day_7",
            "n_agents": len(d),
            "risk_day2_pct": 100.0 * float(d["exposed_by_day2"].mean()),
            "risk_day3_pct": 100.0 * float(d["exposed_by_day3"].mean()),
            "risk_day7_pct": 100.0 * float(d["exposed_by_day7"].mean()),
            "median_time_to_exposure_days": float(d.loc[d["exposure_tick"].le(_day_end_tick(7)), "time_to_exposure_days"].median()) if d["exposure_tick"].le(_day_end_tick(7)).any() else np.nan,
            "mean_trip_steps": float(d["total_trip_steps"].mean()),
            "mean_distance_km": float(d["total_distance_m"].mean() / 1000.0),
        })

    specs = [
        ("student", "School + tap on same day", lambda d: d["school_tap_same_day"]),
        ("student", "School without same-day tap", lambda d: d["visited_school"] & ~d["school_tap_same_day"]),
        ("trader", "Market + tap on same day", lambda d: d["market_tap_same_day"]),
        ("trader", "Market without same-day tap", lambda d: d["visited_market"] & ~d["market_tap_same_day"]),
        ("general", "Activity + tap on same day", lambda d: d["activity_tap_same_day"]),
        ("general", "Activity without same-day tap", lambda d: d[["visited_school", "visited_market", "visited_generic_activity"]].any(axis=1) & ~d["activity_tap_same_day"]),
    ]
    for horizon in (2, 3, 7):
        sus = horizon_tables[horizon]
        sus = sus[sus["initial_state"].eq("S")].copy()
        outcome_col = f"exposed_by_day{horizon}"
        for role, label, mask_fn in specs:
            mask = mask_fn(sus)
            d = sus[sus["role"].eq(role) & mask].copy()
            if d.empty:
                continue
            row = {
                "scenario": scenario, "seed": seed, "role": role, "risk_cohort": label,
                "analysis_type": "role_specific", "horizon_day": horizon, "membership_horizon": f"day_{horizon}",
                "n_agents": len(d), "risk_day2_pct": np.nan, "risk_day3_pct": np.nan, "risk_day7_pct": np.nan,
                "median_time_to_exposure_days": np.nan,
                "mean_trip_steps": float(d["total_trip_steps"].mean()),
                "mean_distance_km": float(d["total_distance_m"].mean() / 1000.0),
            }
            row[f"risk_day{horizon}_pct"] = 100.0 * float(d[outcome_col].mean())
            if horizon in (3, 7):
                early = d[d["exposure_tick"].le(_day_end_tick(horizon))]
                row["median_time_to_exposure_days"] = float(early["time_to_exposure_days"].median()) if not early.empty else np.nan
            role_rows.append(row)

    agent_cols = [
        "scenario", "seed", "agent", "role", "initial_state", "risk_cohort_day3", "risk_cohort_day7",
        "total_trip_steps", "total_distance_m", "became_exposed", "exposed_by_day1", "exposed_by_day2", "exposed_by_day3",
        "exposed_by_day7", "exposure_tick", "exposure_day", "time_to_exposure_ticks", "time_to_exposure_days", "exposure_source",
        "visited_school", "visited_market", "visited_generic_activity", "visited_tap", "visited_toilet",
        "school_tap_same_day", "market_tap_same_day", "activity_tap_same_day",
        "activity_wash_same_day", "activity_to_tap_within_6h",
    ]
    for col in agent_cols:
        if col not in susceptible7.columns:
            susceptible7[col] = np.nan
    return broad_rows, role_rows, susceptible7[agent_cols].copy()


def _plot_role_specific_agent_risk(tables):
    """Role-specific, horizon-correct school/market/activity co-visitation risk plots."""
    rr = tables.get("role_risk", pd.DataFrame()).copy()
    if rr.empty or "analysis_type" not in rr.columns:
        return
    rr = rr[rr["analysis_type"].eq("role_specific")].copy()
    if rr.empty:
        return
    desired = [
        ("student", "School + tap on same day"),
        ("student", "School without same-day tap"),
        ("trader", "Market + tap on same day"),
        ("trader", "Market without same-day tap"),
        ("general", "Activity + tap on same day"),
        ("general", "Activity without same-day tap"),
    ]
    rr["cohort_label"] = rr.apply(
        lambda row: f"{str(row['role']).title()}: {COHORT_LABELS.get(str(row['risk_cohort']), str(row['risk_cohort']))}", axis=1
    )
    order = [f"{role.title()}: {COHORT_LABELS.get(label, label)}" for role, label in desired]
    specs = [
        (2, "risk_day2_pct", "Exposure by day 2", "% exposed", "Reds", ".1f", "supp_role_risk_A_day2"),
        (3, "risk_day3_pct", "Exposure by day 3", "% exposed", "Reds", ".1f", "supp_role_risk_B_day3"),
        (3, "median_time_to_exposure_days", "Median time to early exposure", "Days", "viridis", ".2f", "supp_role_risk_C_time_to_exposure"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
    any_panel = False
    for ax, (horizon, metric, title, cbar, cmap, fmt, stem) in zip(axs, specs):
        d = rr[rr.get("horizon_day", pd.Series(index=rr.index, dtype=float)).eq(horizon)].copy()
        d = d[d["cohort_label"].isin(order)].copy()
        if d.empty:
            ax.axis("off"); continue
        nmean = d.groupby("cohort_label")["n_agents"].mean()
        keep = [x for x in order if x in nmean.index and nmean.loc[x] >= MIN_COHORT_N_FOR_MAIN_PLOT]
        d = d[d["cohort_label"].isin(keep)].copy()
        if d.empty:
            ax.axis("off"); continue
        m, s, ngrid, agg = _scenario_feature_heatmap_data_with_n(d, "cohort_label", metric, "n_agents", keep)
        _save_refined_panel_data(stem, agg, "supplement")
        _heatmap_mean_sd_n_ax(ax, m, s, ngrid, title, cbar, cmap=cmap, fmt=fmt)
        _save_individual_heatmap_panel(stem, m, s, title, cbar, cmap, fmt, "supplement", ngrid, note=f"Membership window: day {horizon}; visits are no later than first exposure. n≈ is mean cohort size. " + _seed_note())
        any_panel = True
    if any_panel:
        fig.suptitle("Role-specific agent risk after experienced activity–tap co-occurrence")
        fig.text(0.5, 0.01, "Each early outcome uses its own visit-membership window. These are temporal same-day visit cohorts, not physical-distance categories.", ha="center", fontsize=8)
        fig.savefig(_paper_path_refined("supp_role_specific_agent_risk", "supplement"), dpi=320, bbox_inches="tight")
    plt.close(fig)


# Role-specific horizon membership changes checkpoint/table schema once more.
COOCCURRENCE_CHECKPOINT_VERSION = 7

_run_parameter_screening()
COOCCURRENCE_TABLES = None
COOCCURRENCE_LAYOUT_RESULTS = None
if RUN_COOCCURRENCE_ENSEMBLE:
    COOCCURRENCE_TABLES, COOCCURRENCE_LAYOUT_RESULTS = run_cooccurrence_ensemble()


# ---- Optional Sobol sensitivity analysis LAST --------------------------------
# Kept at the very end so all old baseline/B/D plots, diagnostics, and tables are
# already saved before the expensive Sobol loop starts.
SOBOL_RESULTS = {}
if not RUN_SOBOL:
    print("[Sobol] skipped because RUN_SOBOL=False. Set RUN_SOBOL=True or run with RUN_SOBOL=1 to generate Sobol CSVs/figures.")
if RUN_SOBOL:
    SOBOL_OUT = Path(CITY_OUT) / "sobol"
    SOBOL_OUT.mkdir(parents=True, exist_ok=True)
    print(f"[Sobol] writing outputs → {SOBOL_OUT}")
    if RUN_SOBOL_MOBILITY:
        SOBOL_RESULTS["mobility"] = _run_and_save_sobol_block(
            "mobility",
            SOBOL_MOBILITY_PROBLEM,
            _build_mobility_sobol_intervention,
            SOBOL_OUT,
        )
    if RUN_SOBOL_DECENTRALISATION:
        SOBOL_RESULTS["decentralisation_activity"] = _run_and_save_sobol_block(
            "decentralisation_activity",
            SOBOL_DECENTRALISATION_ACTIVITY_PROBLEM,
            _build_decentralisation_activity_sobol_intervention,
            SOBOL_OUT,
        )
        SOBOL_RESULTS["decentralisation_wash"] = _run_and_save_sobol_block(
            "decentralisation_wash",
            SOBOL_DECENTRALISATION_WASH_PROBLEM,
            _build_decentralisation_wash_sobol_intervention,
            SOBOL_OUT,
        )
        SOBOL_RESULTS["decentralisation_all"] = _run_and_save_sobol_block(
            "decentralisation_all",
            SOBOL_DECENTRALISATION_ALL_PROBLEM,
            _build_decentralisation_all_sobol_intervention,
            SOBOL_OUT,
        )
    if RUN_SOBOL_FULL:
        SOBOL_RESULTS["full"] = _run_and_save_sobol_block(
            "full",
            SOBOL_FULL_PROBLEM,
            _build_full_sobol_intervention,
            SOBOL_OUT,
        )
    if RUN_SOBOL_DECENTRALISATION_EXTENDED:
        SOBOL_RESULTS["decentralisation_extended"] = _run_and_save_sobol_block(
            "decentralisation_extended",
            SOBOL_DECENTRALISATION_EXTENDED_PROBLEM,
            _build_decentralisation_extended_sobol_intervention,
            SOBOL_OUT,
        )
    print(f"[Sobol] complete. Saved CSVs/figures in {SOBOL_OUT}")

