#!/usr/bin/env python3
"""
Edmonton Canopy Cover Analysis from 2019 LiDAR
================================================
Calculates city-wide tree canopy coverage using height-normalized
vegetation classification rasterization.

Data: 2025 City of Edmonton LiDAR (summer 2025, leaf-on)
Method: USFS i-Tree / American Forests standard approach
"""

import os
import sys
import csv
import json
import time
import glob
import logging
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import binned_statistic_2d
from scipy.ndimage import distance_transform_edt

import laspy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAZ_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR/edmonton-veglidar-laz"
BOUNDARY_FILE = "/tmp/edmonton_boundary_3776.geojson"
OUTPUT_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "canopy_tile_results.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "canopy_cover_report.txt")

CELL_SIZE = 1.0          # metres — raster resolution
HEIGHT_THRESHOLD = 2.0   # metres — minimum height for "tree canopy"
VEG_CLASSES = [3, 4, 5]  # Low, Medium, High Vegetation
GROUND_CLASS = 2

# Height class breakpoints (metres above ground)
LOW_MIN, LOW_MAX = 2.0, 5.0
MED_MIN, MED_MAX = 5.0, 15.0
HIGH_MIN = 15.0

MAX_WORKERS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("canopy")


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------
def load_boundary_polygon(path):
    """Load the Edmonton boundary as a shapely Polygon in EPSG:3776."""
    from shapely.geometry import shape
    with open(path) as f:
        data = json.load(f)
    feat = data["features"][0]
    return shape(feat["geometry"])


def tile_overlaps_boundary(xmin, ymin, xmax, ymax, boundary):
    """Check if a tile bounding box overlaps the city boundary."""
    from shapely.geometry import box
    tile_box = box(xmin, ymin, xmax, ymax)
    return boundary.intersects(tile_box)


def tile_boundary_area(xmin, ymin, xmax, ymax, boundary, cell_size):
    """
    Return a boolean mask (ny, nx) indicating which cells fall inside
    the city boundary. Uses cell centroids for the test.
    """
    from shapely.geometry import box
    from shapely import prepared

    tile_box = box(xmin, ymin, xmax, ymax)
    intersection = boundary.intersection(tile_box)

    nx = int((xmax - xmin) / cell_size)
    ny = int((ymax - ymin) / cell_size)

    if intersection.is_empty:
        return np.zeros((ny, nx), dtype=bool)

    # If tile is fully inside, all cells are valid
    if boundary.contains(tile_box):
        return np.ones((ny, nx), dtype=bool)

    # Otherwise, rasterize the intersection
    from shapely import contains_xy
    cx = np.arange(xmin + cell_size / 2, xmax, cell_size)
    cy = np.arange(ymin + cell_size / 2, ymax, cell_size)
    gx, gy = np.meshgrid(cx, cy)
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    mask = contains_xy(intersection, coords[:, 0], coords[:, 1]).reshape(ny, nx)
    return mask


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------
def process_tile(laz_path, boundary=None):
    """
    Process a single LAZ tile and return canopy statistics.

    Returns dict with:
        tile, total_cells, boundary_cells, canopy_cells,
        low_cells, med_cells, high_cells, xmin, ymin, xmax, ymax
    """
    tile_name = Path(laz_path).stem

    try:
        las = laspy.read(laz_path)
    except Exception as e:
        log.warning(f"Failed to read {tile_name}: {e}")
        return None

    if len(las.points) < 100:
        log.warning(f"Skipping {tile_name}: only {len(las.points)} points")
        return None

    # Tile bounds (snap to integer metres)
    xmin = int(np.floor(las.x.min()))
    xmax = int(np.ceil(las.x.max()))
    ymin = int(np.floor(las.y.min()))
    ymax = int(np.ceil(las.y.max()))
    nx = int((xmax - xmin) / CELL_SIZE)
    ny = int((ymax - ymin) / CELL_SIZE)

    if nx == 0 or ny == 0:
        return None

    # ---- Ground DEM (binned mean) ----
    ground_mask = las.classification == GROUND_CLASS
    gx, gy, gz = las.x[ground_mask], las.y[ground_mask], las.z[ground_mask]

    if len(gx) < 50:
        log.warning(f"Skipping {tile_name}: insufficient ground points ({len(gx)})")
        return None

    x_edges = np.arange(xmin, xmax + CELL_SIZE, CELL_SIZE)
    y_edges = np.arange(ymin, ymax + CELL_SIZE, CELL_SIZE)

    dem, _, _, _ = binned_statistic_2d(
        gx, gy, gz, statistic="mean", bins=[x_edges, y_edges]
    )
    dem = dem.T  # shape: (ny, nx)

    # Fill NaN cells with nearest valid neighbour
    nan_mask = np.isnan(dem)
    if nan_mask.all():
        log.warning(f"Skipping {tile_name}: no valid DEM cells")
        return None
    if nan_mask.any():
        indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        dem = dem[tuple(indices)]

    # ---- Vegetation processing ----
    veg_mask = np.isin(las.classification, VEG_CLASSES)
    vx, vy, vz = las.x[veg_mask], las.y[veg_mask], las.z[veg_mask]

    # Cell indices for vegetation points
    vi = ((vx - xmin) / CELL_SIZE).astype(int)
    vj = ((vy - ymin) / CELL_SIZE).astype(int)
    valid = (vi >= 0) & (vi < nx) & (vj >= 0) & (vj < ny)
    vi, vj, vz = vi[valid], vj[valid], vz[valid]

    # Height above ground
    ground_z = dem[vj, vi]
    hag = vz - ground_z

    # Canopy mask
    canopy = hag >= HEIGHT_THRESHOLD

    # ---- Rasterize ----
    canopy_grid = np.zeros((ny, nx), dtype=np.uint8)
    low_grid = np.zeros((ny, nx), dtype=np.uint8)
    med_grid = np.zeros((ny, nx), dtype=np.uint8)
    high_grid = np.zeros((ny, nx), dtype=np.uint8)

    ci, cj = vi[canopy], vj[canopy]
    heights = hag[canopy]

    if len(ci) > 0:
        canopy_grid[cj, ci] = 1

        low_m = (heights >= LOW_MIN) & (heights < LOW_MAX)
        med_m = (heights >= MED_MIN) & (heights < MED_MAX)
        high_m = heights >= HIGH_MIN

        if low_m.any():
            low_grid[cj[low_m], ci[low_m]] = 1
        if med_m.any():
            med_grid[cj[med_m], ci[med_m]] = 1
        if high_m.any():
            high_grid[cj[high_m], ci[high_m]] = 1

    # ---- Boundary clipping ----
    if boundary is not None:
        try:
            bmask = tile_boundary_area(xmin, ymin, xmax, ymax, boundary, CELL_SIZE)
        except Exception:
            bmask = np.ones((ny, nx), dtype=bool)
    else:
        bmask = np.ones((ny, nx), dtype=bool)

    boundary_cells = int(np.sum(bmask))
    canopy_cells = int(np.sum(canopy_grid & bmask))
    low_cells = int(np.sum(low_grid & bmask))
    med_cells = int(np.sum(med_grid & bmask))
    high_cells = int(np.sum(high_grid & bmask))

    return {
        "tile": tile_name,
        "total_cells": nx * ny,
        "boundary_cells": boundary_cells,
        "canopy_cells": canopy_cells,
        "low_cells": low_cells,
        "med_cells": med_cells,
        "high_cells": high_cells,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


# ---------------------------------------------------------------------------
# Batch processing wrapper (for multiprocessing)
# ---------------------------------------------------------------------------
# Global boundary object is set once per worker via initializer
_boundary = None

def _init_worker(boundary_path):
    global _boundary
    _boundary = load_boundary_polygon(boundary_path)

def _process_tile_wrapper(laz_path):
    return process_tile(laz_path, boundary=_boundary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Edmonton Canopy Cover Analysis")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed tiles")
    parser.add_argument("--limit", type=int, default=0, help="Process only N tiles (0=all)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Edmonton Canopy Cover Analysis")
    log.info("=" * 60)

    # Discover tiles
    laz_files = sorted(glob.glob(os.path.join(LAZ_DIR, "*.laz")))
    log.info(f"Found {len(laz_files)} LAZ tiles in {LAZ_DIR}")

    if args.limit > 0:
        laz_files = laz_files[: args.limit]
        log.info(f"Limiting to {args.limit} tiles")

    # Load already-processed tiles for resume
    done_tiles = set()
    if args.resume and os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_tiles.add(row["tile"])
        log.info(f"Resuming: {len(done_tiles)} tiles already processed")
        laz_files = [f for f in laz_files if Path(f).stem not in done_tiles]
        log.info(f"Remaining: {len(laz_files)} tiles to process")

    if not laz_files:
        log.info("No tiles to process.")
        return

    # Verify boundary file
    if not os.path.exists(BOUNDARY_FILE):
        log.error(f"Boundary file not found: {BOUNDARY_FILE}")
        sys.exit(1)

    boundary = load_boundary_polygon(BOUNDARY_FILE)
    log.info(f"Boundary area: {boundary.area / 1e6:.1f} km²")

    # Open CSV for writing
    write_header = not (args.resume and os.path.exists(RESULTS_CSV))
    csv_mode = "w" if write_header else "a"
    fieldnames = [
        "tile", "total_cells", "boundary_cells", "canopy_cells",
        "low_cells", "med_cells", "high_cells",
        "xmin", "ymin", "xmax", "ymax",
    ]

    csvfile = open(RESULTS_CSV, csv_mode, newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    # Process
    t_start = time.time()
    processed = 0
    errors = 0

    log.info(f"Processing {len(laz_files)} tiles with {args.workers} workers...")

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(BOUNDARY_FILE,),
    ) as executor:
        futures = {executor.submit(_process_tile_wrapper, f): f for f in laz_files}

        for future in as_completed(futures):
            laz_path = futures[future]
            try:
                result = future.result()
                if result is not None:
                    writer.writerow(result)
                    csvfile.flush()
                    processed += 1
                else:
                    errors += 1
            except Exception as e:
                log.error(f"Error processing {Path(laz_path).stem}: {e}")
                errors += 1

            total_done = processed + errors
            if total_done % 100 == 0:
                elapsed = time.time() - t_start
                rate = total_done / elapsed
                remaining = (len(laz_files) - total_done) / rate if rate > 0 else 0
                log.info(
                    f"Progress: {total_done}/{len(laz_files)} "
                    f"({processed} OK, {errors} err) "
                    f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]"
                )

    csvfile.close()
    elapsed = time.time() - t_start
    log.info(f"Done: {processed} tiles in {elapsed:.1f}s ({errors} errors)")

    # ---- Generate report ----
    generate_report()


def generate_report():
    """Read the CSV results and produce a summary report."""
    if not os.path.exists(RESULTS_CSV):
        log.error("No results CSV found")
        return

    rows = []
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_boundary_cells = sum(int(r["boundary_cells"]) for r in rows)
    total_canopy_cells = sum(int(r["canopy_cells"]) for r in rows)
    total_low = sum(int(r["low_cells"]) for r in rows)
    total_med = sum(int(r["med_cells"]) for r in rows)
    total_high = sum(int(r["high_cells"]) for r in rows)

    # Each cell is CELL_SIZE × CELL_SIZE m²
    cell_area_m2 = CELL_SIZE * CELL_SIZE
    boundary_area_m2 = total_boundary_cells * cell_area_m2
    canopy_area_m2 = total_canopy_cells * cell_area_m2

    boundary_area_km2 = boundary_area_m2 / 1e6
    boundary_area_ha = boundary_area_m2 / 1e4
    canopy_area_km2 = canopy_area_m2 / 1e6
    canopy_area_ha = canopy_area_m2 / 1e4

    canopy_pct = 100 * total_canopy_cells / total_boundary_cells if total_boundary_cells > 0 else 0

    low_pct = 100 * total_low / total_boundary_cells if total_boundary_cells > 0 else 0
    med_pct = 100 * total_med / total_boundary_cells if total_boundary_cells > 0 else 0
    high_pct = 100 * total_high / total_boundary_cells if total_boundary_cells > 0 else 0

    report = f"""
================================================================================
  EDMONTON CANOPY COVER ANALYSIS — 2025 LiDAR
================================================================================

  Data Source:    2025 City of Edmonton LiDAR Data Collection
  Collection:     Summer 2025, leaf-on condition
  Point Density:  25 pts/m², 20 cm nominal spacing
  CRS:            EPSG:3776 (NAD83 / Alberta 3TM ref merid 114 W)
  Method:         Height-normalized vegetation rasterization (1m grid)
  Canopy Def:     Vegetation points (classes 3/4/5) ≥ {HEIGHT_THRESHOLD}m above ground

  Tiles Processed:  {len(rows):,}

  RESULTS
  -------
  City Land Area:       {boundary_area_km2:,.1f} km²  ({boundary_area_ha:,.0f} ha)
  Total Canopy Area:    {canopy_area_km2:,.1f} km²  ({canopy_area_ha:,.0f} ha)

  ┌──────────────────────────────────────────────┐
  │  CANOPY COVER:  {canopy_pct:.1f}%                        │
  └──────────────────────────────────────────────┘

  Height Class Breakdown:
    Low canopy   (2–5 m):   {low_pct:.1f}%   ({total_low * cell_area_m2 / 1e4:,.0f} ha)
    Medium canopy (5–15 m):  {med_pct:.1f}%   ({total_med * cell_area_m2 / 1e4:,.0f} ha)
    High canopy  (>15 m):    {high_pct:.1f}%   ({total_high * cell_area_m2 / 1e4:,.0f} ha)

  Notes:
  - Canopy cover = canopy cells / total boundary cells at {CELL_SIZE}m resolution
  - Height classes may overlap spatially (a cell with both 4m and 12m points
    counts in both low and medium), so height-class percentages may sum > total
  - Boundary from OpenStreetMap (Nominatim), reprojected to EPSG:3776
  - "Vegetation" = LAS classes 3 (Low), 4 (Medium), 5 (High) only
    Class 1 (Unclassified) excluded to avoid counting buildings

================================================================================
"""

    with open(REPORT_FILE, "w") as f:
        f.write(report)

    print(report)
    log.info(f"Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
