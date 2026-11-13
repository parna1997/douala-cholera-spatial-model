#!/usr/bin/env python3
"""Publication Figure 6(c-f): readable light-background spatial maps.

Uses saved 50-seed spatial peak E+I arrays and the exact matching scenario
layouts. No simulation is executed and no input data are modified.

Layers (bottom to top):
  white background; warm amber generic-POI density shading; translucent blue
  E+I density; bold city and hub outlines; small, contrasting facility symbols.

Example:
  python render_figure6_maps.py \
    --run-dir outputs/douala/run_3000/cooccurrence_ensemble \
    --out-dir outputs/douala/run_3000/publication_figure6
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import gaussian_filter

SCENARIOS = (
    ("baseline", "Baseline", "fig6C"),
    ("intervD_destinations_only", "POI", "fig6D"),
    ("intervD_facilities_only", "WASH", "fig6E"),
    ("intervD_area_redistribute", "POI+WASH", "fig6F"),
)

# The exposure colourbar is consistent across all four scenarios. The PowerNorm
# transform helps reveal the diffuse low-density areas without changing values.
EXPOSURE_CMAP = LinearSegmentedColormap.from_list("exposure_blues", [
    (0.00, "#ffffff"), (0.04, "#d9f1ff"), (0.16, "#9ed4f0"),
    (0.34, "#4caadc"), (0.58, "#1681bf"), (0.80, "#075ca0"),
    (1.00, "#083970"),
], N=512)

# Marker area is measured in points squared: deliberately small and not outlined
# to avoid obscuring fine spatial details.
FACILITIES = (
    ("schools_active", "^", 34, "#93278f", "Schools"),
    ("markets_active", "s", 28, "#e67e22", "Markets"),
    ("taps_active", "o", 29, "#06a9d0", "Water taps"),
    ("toilets_active", "x", 40, "#191919", "Public toilets"),
)


def load_checkpoint(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint missing: {path}")
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def layer(layout: dict, name: str):
    return layout.get(name) if isinstance(layout, dict) else None


def present(gdf) -> bool:
    return gdf is not None and not getattr(gdf, "empty", True)


def extent_from_layout(layout: dict):
    area = layer(layout, "outer_cells_active")
    if not present(area):
        area = layer(layout, "inner_hubs_active")
    if not present(area):
        raise ValueError("Layout has no city or hub geometry")
    xmin, ymin, xmax, ymax = map(float, area.total_bounds)
    return (xmin, xmax, ymin, ymax)


def generic_density(layout: dict, extent, shape, sigma: float):
    """Smoothed generic POI counts per map cell (not invented POI locations)."""
    g = layer(layout, "generic_active")
    if not present(g):
        raise ValueError("'generic_active' missing or empty in layout; no density shading can be drawn")
    xs = np.asarray(g.geometry.x, dtype=float)
    ys = np.asarray(g.geometry.y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if not len(xs):
        raise ValueError("Generic POI geometries contain no valid points")
    ny, nx = shape
    xmin, xmax, ymin, ymax = extent
    # Histogram uses (y, x) bin order to match an origin='lower' raster.
    counts, _, _ = np.histogram2d(ys, xs, bins=(ny, nx), range=((ymin, ymax), (xmin, xmax)))
    smooth = gaussian_filter(counts, sigma=sigma, mode="constant")
    return smooth, int(len(xs))


def density_rgba(density, common_max):
    """Warm beige/amber shading, transparent where there are no nearby POIs."""
    intensity = np.clip(density / max(common_max, 1e-12), 0.0, 1.0)
    # A local smooth region occupies several cells; lower but nonzero values
    # remain visible, while cells outside each density patch stay transparent.
    alpha = np.where(intensity > 0.035, 0.18 + 0.58 * intensity ** 0.55, 0.0)
    rgba = np.empty((*density.shape, 4), dtype=float)
    rgba[..., 0] = 0.86  # muted gold-brown for clear distinction from blue exposure
    rgba[..., 1] = 0.60
    rgba[..., 2] = 0.23
    rgba[..., 3] = alpha
    return rgba


def exposure_rgba(arr, norm):
    """Blue E+I shading with fully transparent zeros; POI layer shows through."""
    safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = np.clip(np.asarray(norm(safe), dtype=float), 0.0, 1.0)
    rgba = EXPOSURE_CMAP(scaled)
    rgba[..., 3] = np.where(safe > 0.0, 0.44 + 0.46 * scaled ** 0.42, 0.0)
    return rgba


def draw_markers(ax, layout):
    for name, marker, area, colour, label in FACILITIES:
        pts = layer(layout, name)
        if not present(pts):
            print(f"  note: layout has no {name} points", flush=True)
            continue
        xs, ys = pts.geometry.x, pts.geometry.y
        if marker == "x":
            # Single dark cross; avoid broad white or black halos.
            ax.scatter(xs, ys, marker="x", s=area, c=colour,
                       linewidths=1.55, zorder=16)
        else:
            ax.scatter(xs, ys, marker=marker, s=area, c=colour,
                       edgecolors="#ffffff", linewidths=0.8, zorder=16)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed-start", default=42, type=int)
    p.add_argument("--n-seeds", default=50, type=int)
    p.add_argument("--checkpoint-version", default=7, type=int)
    p.add_argument("--poi-sigma", default=3.2, type=float,
                   help="Smoothing of the actual generic POI point-density grid (map pixels)")
    p.add_argument("--exposure-gamma", default=0.52, type=float,
                   help="Colour normalisation gamma below 1 boosts lower exposure densities")
    p.add_argument("--dpi", default=400, type=int)
    args = p.parse_args()
    if args.poi_sigma <= 0 or args.exposure_gamma <= 0:
        p.error("--poi-sigma and --exposure-gamma must be positive")

    ckdir = args.run_dir / "checkpoints"
    if not ckdir.is_dir():
        raise FileNotFoundError(f"Checkpoints folder not found: {ckdir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    layouts, means, poi_grids, summaries = {}, {}, {}, []
    vmax, poi_max = 0.0, 0.0
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)

    # Load fixed scenario-specific layouts and all 50 archived spatial arrays.
    # This computation is identical to the prior mean-of-peak_ei calculation.
    for scenario, label, stem in SCENARIOS:
        layout_data = load_checkpoint(ckdir / f"v{args.checkpoint_version}_layout_{scenario}.pkl.gz")
        layouts[scenario] = layout_data
        arrays = []
        for seed in seeds:
            path = ckdir / f"v{args.checkpoint_version}_seed_{seed:03d}_{scenario}.pkl.gz"
            data = load_checkpoint(path)
            arr = np.asarray(data["spatial"]["peak_ei"], dtype=float)
            if arr.ndim != 2 or (arrays and arr.shape != arrays[0].shape):
                raise ValueError(f"Invalid/inconsistent peak_ei array shape in {path}")
            arrays.append(arr)
        mean_arr = np.stack(arrays).mean(axis=0)
        means[scenario] = mean_arr
        vmax = max(vmax, float(np.nanmax(mean_arr)))
        poi_grid, count = generic_density(
            layout_data, extent_from_layout(layout_data), mean_arr.shape, args.poi_sigma
        )
        poi_grids[scenario] = poi_grid
        poi_max = max(poi_max, float(poi_grid.max()))
        summaries.append({
            "scenario": label, "seeds": len(arrays),
            "peak_ei_array_shape": list(mean_arr.shape),
            "peak_ei_max": float(mean_arr.max()),
            "generic_poi_points": count,
            "generic_poi_smoothed_max": float(poi_grid.max()),
        })
        print(f"{label}: {len(arrays)} checkpoint arrays; {count} generic POI locations; "
              f"peak E+I max {mean_arr.max():.4g}", flush=True)

    if not np.isfinite(vmax) or vmax <= 0 or poi_max <= 0:
        raise ValueError("Expected positive E+I density and generic POI density")
    norm = PowerNorm(gamma=args.exposure_gamma, vmin=0.0, vmax=vmax, clip=True)

    for scenario, label, stem in SCENARIOS:
        layout_data = layouts[scenario]
        extent = extent_from_layout(layout_data)
        fig, ax = plt.subplots(figsize=(10.5, 9.2), facecolor="white")
        ax.set_facecolor("white")

        # Two independent map layers: warm shaded POI area AND blue E+I density.
        ax.imshow(density_rgba(poi_grids[scenario], poi_max), origin="lower",
                  extent=extent, interpolation="bilinear", zorder=2)
        ax.imshow(exposure_rgba(means[scenario], norm), origin="lower",
                  extent=extent, interpolation="nearest", zorder=4)

        # Bold ONLY city/hub polygon boundaries; not the markers.
        outer = layer(layout_data, "outer_cells_active")
        hubs = layer(layout_data, "inner_hubs_active")
        if present(outer):
            outer.boundary.plot(ax=ax, color="#142537", linewidth=2.4, zorder=12)
        if present(hubs):
            hubs.boundary.plot(ax=ax, color="#384b5e", linewidth=1.55, zorder=13)
        draw_markers(ax, layout_data)

        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box"); ax.axis("off")
        fig.tight_layout(pad=0.08)
        for suffix in ("png", "pdf"):
            fig.savefig(args.out_dir / f"{stem}.{suffix}", dpi=args.dpi,
                        bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
        print(f"Saved {stem}.png and {stem}.pdf", flush=True)

    # Unified marker key, including the new generic density layer.
    legend_handles = [Patch(facecolor="#db993a", alpha=0.66,
                            label="Generic POI density", edgecolor="none")]
    for _, mk, _, colour, label in FACILITIES:
        if mk == "x":
            legend_handles.append(Line2D([0], [0], marker=mk, linestyle="None",
                                         color=colour, markeredgewidth=1.5,
                                         markersize=7, label=label))
        else:
            legend_handles.append(Line2D([0], [0], marker=mk, linestyle="None",
                                         markerfacecolor=colour, markeredgecolor="white",
                                         color="none", markersize=7, label=label))
    fig, ax = plt.subplots(figsize=(11.7, 1.15), facecolor="white")
    ax.axis("off")
    ax.legend(handles=legend_handles, ncol=5, loc="center", frameon=False,
              fontsize=12, handletextpad=0.35, columnspacing=1.25)
    fig.savefig(args.out_dir / "fig6_legend.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(args.out_dir / "fig6_legend.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    # Shared colorbar. PowerNorm is applied consistently to every map.
    fig, ax = plt.subplots(figsize=(9.5, 1.1), facecolor="white")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=EXPOSURE_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.ax.tick_params(labelsize=12, width=1.1)
    cb.outline.set_linewidth(1.15)
    cb.set_label("Mean E + I density at scenario-specific peak", size=12, labelpad=8)
    fig.savefig(args.out_dir / "fig6_colorbar.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(args.out_dir / "fig6_colorbar.pdf", bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)

    provenance = {
        "input_directory": str(args.run_dir),
        "checkpoint_version": args.checkpoint_version,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "shared_ei_density_max": vmax,
        "shared_generic_poi_smoothed_max": poi_max,
        "poi_smoothing_sigma_grid_cells": args.poi_sigma,
        "exposure_powernorm_gamma": args.exposure_gamma,
        "exposure_colourbar": "shared PowerNorm; blue, transparent for zero values",
        "generic_poi_density": "smoothed histogram of archived generic_active POI locations; amber transparency",
        "calculation": "mean of archived peak_ei arrays per scenario; plotting only; no simulation rerun",
        "scenarios": summaries,
    }
    (args.out_dir / "figure6_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("DONE: Figure 6(c-f), amber generic POI areas, blue exposure density, compact symbols.", flush=True)


if __name__ == "__main__":
    main()
