#!/usr/bin/env python3
"""Create the model's WGS84 x,y,z CSV from a downloaded WorldPop GeoTIFF.

External data are not redistributed. Supply the 2020 Cameroon WorldPop raster
at inputs/douala/data/pop.tif; this script writes cell-centre coordinates and
positive population values to inputs/douala/data/pop.csv. The model also reads
the original GeoTIFF; neither input is changed by the simulation itself.

Example:
    python code/prepare_population.py \
        --raster inputs/douala/data/pop.tif \
        --csv inputs/douala/data/pop.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def convert(raster: Path, destination: Path, block_rows: int = 512) -> int:
    """Stream the first raster band into x,y,z rows; no country-sized array copy."""
    if raster.resolve() == destination.resolve():
        raise ValueError("Input raster and output CSV paths must be different")
    if not raster.is_file():
        raise FileNotFoundError(f"WorldPop raster missing: {raster}")
    with rasterio.open(raster) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError("The supplied WorldPop raster must use WGS84 (EPSG:4326)")
        if src.count < 1:
            raise ValueError("WorldPop raster has no bands")
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        transform = src.transform
        with destination.open('w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out)
            writer.writerow(['x', 'y', 'z'])
            for row0 in range(0, src.height, block_rows):
                height = min(block_rows, src.height - row0)
                window = Window(0, row0, src.width, height)
                band = src.read(1, window=window, masked=True)
                values = np.asarray(band.filled(np.nan), dtype=float)
                keep = np.isfinite(values) & (values > 0)
                local_rows, local_cols = np.nonzero(keep)
                if len(local_rows) == 0:
                    continue
                cols = local_cols + 0.5
                rows = local_rows + row0 + 0.5
                xs = transform.a * cols + transform.b * rows + transform.c
                ys = transform.d * cols + transform.e * rows + transform.f
                writer.writerows(zip(xs, ys, values[keep]))
                written += len(local_rows)
    if written == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("No positive population cells were found in the raster")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raster', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    rows = convert(args.raster, args.csv)
    print(f"Saved {rows:,} positive-population cells to {args.csv}")


if __name__ == '__main__':
    main()
