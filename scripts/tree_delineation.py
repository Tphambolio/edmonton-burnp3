#!/usr/bin/env python3
"""
Edmonton Individual Tree Delineation from 2025 LiDAR
=====================================================
Identifies individual trees and shrubs from LiDAR point cloud data using:
1. Canopy Height Model (CHM) generation
2. Gaussian smoothing to reduce noise
3. Local maxima detection for tree tops
4. Watershed segmentation for crown delineation
5. Per-crown feature extraction (height, area, shape)
6. Optional: conifer/deciduous classification using RGB ortho + LiDAR features

Data: 2025 City of Edmonton LiDAR (summer 2025, leaf-on)
"""

import os
import csv
import json
import time
import glob
import logging
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import laspy
from scipy.stats import binned_statistic_2d
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label as ndimage_label,
    maximum_filter,
)
from skimage.segmentation import watershed
from skimage.measure import regionprops

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trees")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAZ_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR/edmonton-veglidar-laz"
ORTHO_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR/ortho_tiles"
OUTPUT_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR/tree_results"
TREES_CSV = os.path.join(OUTPUT_DIR, "individual_trees.csv")

CELL_SIZE = 0.5           # metres — CHM resolution (Challenger uses 0.5m)
HEIGHT_THRESHOLD = 2.0    # metres — minimum height for tree detection
SHRUB_MAX_HEIGHT = 5.0    # metres — shrub vs tree boundary
SMOOTHING_SIGMA = 1.5     # Gaussian sigma for CHM smoothing (in cells)
MIN_CROWN_AREA = 2.0      # m² — minimum crown area to keep
MIN_TREE_SPACING = 2.0    # metres — minimum distance between tree tops

VEG_CLASSES = [3, 4, 5]
GROUND_CLASS = 2
MAX_WORKERS = 8


# ---------------------------------------------------------------------------
# CHM Generation
# ---------------------------------------------------------------------------
def build_chm(las, cell_size=CELL_SIZE):
    """
    Build a Canopy Height Model from LiDAR points.
    Returns: chm (2D array), dem (2D array), xmin, ymin, nx, ny
    """
    xmin = int(np.floor(las.x.min()))
    xmax = int(np.ceil(las.x.max()))
    ymin = int(np.floor(las.y.min()))
    ymax = int(np.ceil(las.y.max()))
    nx = int((xmax - xmin) / cell_size)
    ny = int((ymax - ymin) / cell_size)

    x_edges = np.arange(xmin, xmax + cell_size, cell_size)
    y_edges = np.arange(ymin, ymax + cell_size, cell_size)

    # Ground DEM (mean of ground points)
    ground_mask = las.classification == GROUND_CLASS
    gx, gy, gz = las.x[ground_mask], las.y[ground_mask], las.z[ground_mask]

    if len(gx) < 50:
        return None, None, xmin, ymin, nx, ny

    dem, _, _, _ = binned_statistic_2d(gx, gy, gz, statistic="mean", bins=[x_edges, y_edges])
    dem = dem.T

    # Fill NaN in DEM
    nan_mask = np.isnan(dem)
    if nan_mask.all():
        return None, None, xmin, ymin, nx, ny
    if nan_mask.any():
        indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        dem = dem[tuple(indices)]

    # DSM (max elevation of ALL non-ground, non-noise points)
    surface_mask = np.isin(las.classification, VEG_CLASSES) | (las.classification == 1)
    sx, sy, sz = las.x[surface_mask], las.y[surface_mask], las.z[surface_mask]

    if len(sx) < 10:
        return None, dem, xmin, ymin, nx, ny

    dsm, _, _, _ = binned_statistic_2d(sx, sy, sz, statistic="max", bins=[x_edges, y_edges])
    dsm = dsm.T

    # CHM = DSM - DEM
    chm = dsm - dem
    chm[np.isnan(chm)] = 0
    chm[chm < 0] = 0

    return chm, dem, xmin, ymin, nx, ny


# ---------------------------------------------------------------------------
# Tree Top Detection
# ---------------------------------------------------------------------------
def detect_tree_tops(chm, cell_size=CELL_SIZE, min_height=HEIGHT_THRESHOLD,
                     sigma=SMOOTHING_SIGMA, min_spacing=MIN_TREE_SPACING):
    """
    Detect individual tree tops as local maxima in a smoothed CHM.
    Returns: array of (row, col) indices of tree tops
    """
    # Smooth CHM to reduce noise
    chm_smooth = gaussian_filter(chm.astype(np.float64), sigma=sigma)

    # Local maxima detection with minimum spacing
    spacing_cells = max(int(min_spacing / cell_size), 3)
    local_max = maximum_filter(chm_smooth, size=spacing_cells)
    peaks = (chm_smooth == local_max) & (chm_smooth >= min_height)

    # Get peak coordinates
    rows, cols = np.where(peaks)

    if len(rows) == 0:
        return np.array([]), chm_smooth

    # Get heights at peaks
    heights = chm_smooth[rows, cols]

    return np.column_stack([rows, cols, heights]), chm_smooth


# ---------------------------------------------------------------------------
# Crown Delineation via Watershed
# ---------------------------------------------------------------------------
def delineate_crowns(chm_smooth, tree_tops, min_height=HEIGHT_THRESHOLD):
    """
    Delineate individual tree crowns using watershed segmentation.
    Returns: labeled crown array (same shape as chm)
    """
    if len(tree_tops) == 0:
        return np.zeros_like(chm_smooth, dtype=np.int32)

    # Create markers for watershed
    markers = np.zeros_like(chm_smooth, dtype=np.int32)
    for i, (r, c, h) in enumerate(tree_tops, start=1):
        markers[int(r), int(c)] = i

    # Mask: only segment where CHM > threshold
    mask = chm_smooth >= min_height

    # Watershed on inverted CHM (valleys become peaks for watershed)
    # Higher CHM = lower "elevation" for watershed → crowns form around peaks
    inverted = -chm_smooth.copy()

    labels = watershed(inverted, markers=markers, mask=mask)

    return labels


# ---------------------------------------------------------------------------
# Crown Feature Extraction
# ---------------------------------------------------------------------------
def extract_crown_features(chm, chm_smooth, labels, tree_tops, dem,
                           xmin, ymin, cell_size, las=None):
    """
    Extract per-crown features for each delineated tree.
    Returns: list of dicts with tree attributes
    """
    trees = []
    props = regionprops(labels, intensity_image=chm_smooth)

    # Build a lookup from label to tree top
    top_lookup = {}
    for i, (r, c, h) in enumerate(tree_tops, start=1):
        top_lookup[i] = (int(r), int(c), float(h))

    for prop in props:
        lbl = prop.label
        if lbl not in top_lookup:
            continue

        top_r, top_c, top_h = top_lookup[lbl]

        # Crown area
        area_m2 = prop.area * cell_size * cell_size
        if area_m2 < MIN_CROWN_AREA:
            continue

        # Crown centroid (in grid coords → real coords)
        cy, cx = prop.centroid
        real_x = xmin + cx * cell_size + cell_size / 2
        real_y = ymin + cy * cell_size + cell_size / 2

        # Tree top position
        top_x = xmin + top_c * cell_size + cell_size / 2
        top_y = ymin + top_r * cell_size + cell_size / 2

        # Height stats within crown
        crown_mask = labels == lbl
        crown_heights = chm[crown_mask]
        crown_heights = crown_heights[crown_heights > 0]

        if len(crown_heights) == 0:
            continue

        max_height = float(np.max(crown_heights))
        mean_height = float(np.mean(crown_heights))
        std_height = float(np.std(crown_heights))

        # Crown diameter (approximate from area assuming circular)
        diameter = 2 * np.sqrt(area_m2 / np.pi)

        # Shape metrics
        perimeter = prop.perimeter * cell_size
        compactness = (4 * np.pi * area_m2) / (perimeter ** 2) if perimeter > 0 else 0
        eccentricity = prop.eccentricity

        # Ground elevation at tree top
        ground_elev = float(dem[top_r, top_c]) if dem is not None else 0

        # Vegetation class (height-based)
        if max_height < SHRUB_MAX_HEIGHT:
            veg_type = "shrub"
        else:
            veg_type = "tree"

        # LiDAR-based features for conifer/deciduous (if point cloud available)
        # These will be populated later when we have RGB ortho
        tree = {
            "tree_id": lbl,
            "top_x": round(top_x, 2),
            "top_y": round(top_y, 2),
            "centroid_x": round(real_x, 2),
            "centroid_y": round(real_y, 2),
            "ground_elev": round(ground_elev, 2),
            "max_height": round(max_height, 2),
            "mean_height": round(mean_height, 2),
            "std_height": round(std_height, 2),
            "crown_area_m2": round(area_m2, 2),
            "crown_diameter_m": round(diameter, 2),
            "perimeter_m": round(perimeter, 2),
            "compactness": round(compactness, 4),
            "eccentricity": round(eccentricity, 4),
            "veg_type": veg_type,
        }
        trees.append(tree)

    return trees


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------
def process_tile(laz_path):
    """Process a single LAZ tile for individual tree detection."""
    tile_name = Path(laz_path).stem

    try:
        las = laspy.read(laz_path)
    except Exception as e:
        log.warning(f"Failed to read {tile_name}: {e}")
        return None, tile_name

    if len(las.points) < 100:
        return None, tile_name

    # Build CHM
    chm, dem, xmin, ymin, nx, ny = build_chm(las)
    if chm is None:
        log.warning(f"Skipping {tile_name}: no CHM generated")
        return None, tile_name

    # Detect tree tops
    tree_tops, chm_smooth = detect_tree_tops(chm)
    if len(tree_tops) == 0:
        return [], tile_name

    # Delineate crowns
    crown_labels = delineate_crowns(chm_smooth, tree_tops)

    # Extract features
    trees = extract_crown_features(
        chm, chm_smooth, crown_labels, tree_tops, dem,
        xmin, ymin, CELL_SIZE, las
    )

    # Add tile name to each tree
    for t in trees:
        t["tile"] = tile_name

    return trees, tile_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Edmonton Tree Delineation")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=0, help="Process only N tiles (0=all)")
    parser.add_argument("--tiles", nargs="+", help="Specific tile names to process")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("Edmonton Individual Tree Delineation")
    log.info("=" * 60)

    # Discover tiles
    if args.tiles:
        laz_files = [os.path.join(LAZ_DIR, f"{t}.laz") for t in args.tiles]
    else:
        laz_files = sorted(glob.glob(os.path.join(LAZ_DIR, "*.laz")))

    log.info(f"Found {len(laz_files)} LAZ tiles")

    if args.limit > 0:
        laz_files = laz_files[:args.limit]
        log.info(f"Limiting to {args.limit} tiles")

    # Resume support
    done_tiles = set()
    if args.resume and os.path.exists(TREES_CSV):
        with open(TREES_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_tiles.add(row["tile"])
        log.info(f"Resuming: {len(done_tiles)} tiles already processed")
        laz_files = [f for f in laz_files if Path(f).stem not in done_tiles]

    if not laz_files:
        log.info("No tiles to process.")
        return

    # CSV setup
    fieldnames = [
        "tile", "tree_id", "top_x", "top_y", "centroid_x", "centroid_y",
        "ground_elev", "max_height", "mean_height", "std_height",
        "crown_area_m2", "crown_diameter_m", "perimeter_m",
        "compactness", "eccentricity", "veg_type",
    ]

    write_header = not (args.resume and os.path.exists(TREES_CSV))
    csv_mode = "w" if write_header else "a"
    csvfile = open(TREES_CSV, csv_mode, newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    t_start = time.time()
    total_trees = 0
    processed = 0
    errors = 0

    log.info(f"Processing {len(laz_files)} tiles with {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_tile, f): f for f in laz_files}

        for future in as_completed(futures):
            try:
                trees, tile_name = future.result()
                if trees is not None:
                    for t in trees:
                        writer.writerow(t)
                    csvfile.flush()
                    total_trees += len(trees)
                    processed += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                log.error(f"Error: {e}")

            total_done = processed + errors
            if total_done % 50 == 0:
                elapsed = time.time() - t_start
                rate = total_done / elapsed if elapsed > 0 else 0
                remaining = (len(laz_files) - total_done) / rate if rate > 0 else 0
                log.info(
                    f"Progress: {total_done}/{len(laz_files)} tiles "
                    f"({total_trees:,} trees found) "
                    f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]"
                )

    csvfile.close()
    elapsed = time.time() - t_start

    log.info(f"=" * 60)
    log.info(f"Complete: {processed} tiles, {total_trees:,} trees in {elapsed:.1f}s")
    log.info(f"Results: {TREES_CSV}")
    log.info(f"Errors: {errors}")


if __name__ == "__main__":
    main()
