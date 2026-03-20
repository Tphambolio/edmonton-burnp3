#!/usr/bin/env python3
"""
Extract per-crown LiDAR features — memory-safe version
========================================================
Processes tiles in small batches with explicit garbage collection.
Appends to existing CSV for resume support.
"""

import os
import csv
import gc
import time
import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d
from scipy.ndimage import distance_transform_edt

import laspy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feat2")

BASE_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"
LAZ_DIR = os.path.join(BASE_DIR, "edmonton-veglidar-laz")
TREES_CSV = os.path.join(BASE_DIR, "tree_results/trees_classified.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "tree_results/crown_features.csv")

CELL_SIZE = 0.5
VEG_CLASSES = [3, 4, 5]
GROUND_CLASS = 2
MAX_WORKERS = 4  # reduced from 8 to control memory
BATCH_SIZE = 100  # process in batches, gc between


def process_single_tile(tile_name, crown_records):
    """Process one tile: read LAZ, build KD-tree, extract per-crown features."""
    laz_path = os.path.join(LAZ_DIR, f"{tile_name}.laz")
    if not os.path.exists(laz_path):
        return []

    try:
        las = laspy.read(laz_path)
    except Exception:
        return []

    # Ground DEM
    ground_mask = las.classification == GROUND_CLASS
    gx, gy, gz = las.x[ground_mask], las.y[ground_mask], las.z[ground_mask]
    if len(gx) < 50:
        return []

    xmin = int(np.floor(las.x.min()))
    xmax = int(np.ceil(las.x.max()))
    ymin = int(np.floor(las.y.min()))
    ymax = int(np.ceil(las.y.max()))
    x_edges = np.arange(xmin, xmax + CELL_SIZE, CELL_SIZE)
    y_edges = np.arange(ymin, ymax + CELL_SIZE, CELL_SIZE)
    nx = int((xmax - xmin) / CELL_SIZE)
    ny = int((ymax - ymin) / CELL_SIZE)

    dem, _, _, _ = binned_statistic_2d(gx, gy, gz, statistic="mean", bins=[x_edges, y_edges])
    dem = dem.T
    nan_mask = np.isnan(dem)
    if nan_mask.all():
        return []
    if nan_mask.any():
        indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        dem = dem[tuple(indices)]

    # Vegetation points
    veg_mask = np.isin(las.classification, VEG_CLASSES)
    vx = np.asarray(las.x[veg_mask])
    vy = np.asarray(las.y[veg_mask])
    vz = np.asarray(las.z[veg_mask])

    has_rgb = hasattr(las, 'red')
    if has_rgb:
        vr = np.asarray(las.red[veg_mask], dtype=np.float32)
        vg = np.asarray(las.green[veg_mask], dtype=np.float32)
        vb = np.asarray(las.blue[veg_mask], dtype=np.float32)
        if vr.max() > 255:
            vr /= 256.0
            vg /= 256.0
            vb /= 256.0

    has_intensity = hasattr(las, 'intensity')
    if has_intensity:
        v_intensity = np.asarray(las.intensity[veg_mask], dtype=np.float32)

    has_returns = hasattr(las, 'return_number') and hasattr(las, 'number_of_returns')
    if has_returns:
        v_return_num = np.asarray(las.return_number[veg_mask])
        v_num_returns = np.asarray(las.number_of_returns[veg_mask])

    v_class = np.asarray(las.classification[veg_mask])

    # Free LAS object
    del las
    gc.collect()

    # Height above ground
    vi = np.clip(((vx - xmin) / CELL_SIZE).astype(int), 0, nx - 1)
    vj = np.clip(((vy - ymin) / CELL_SIZE).astype(int), 0, ny - 1)
    ground_z = dem[vj, vi]
    hag = vz - ground_z

    # KD-tree
    veg_xy = np.column_stack([vx, vy])
    kdtree = cKDTree(veg_xy)

    results = []
    for rec in crown_records:
        tree_id = rec["tree_id"]
        cx, cy = rec["centroid_x"], rec["centroid_y"]
        radius = rec["crown_diameter_m"] / 2.0

        pt_indices = kdtree.query_ball_point([cx, cy], radius)
        n_pts = len(pt_indices)
        if n_pts < 5:
            continue

        idx = np.array(pt_indices)
        h = hag[idx]
        h_pos = h[h > 0]
        if len(h_pos) < 3:
            continue

        feat = {"tile": tile_name, "tree_id": tree_id, "n_points": int(n_pts)}

        # Height features
        feat["h_max"] = float(np.max(h_pos))
        feat["h_mean"] = float(np.mean(h_pos))
        feat["h_std"] = float(np.std(h_pos))
        feat["h_median"] = float(np.median(h_pos))
        feat["h_p25"] = float(np.percentile(h_pos, 25))
        feat["h_p75"] = float(np.percentile(h_pos, 75))
        feat["h_p95"] = float(np.percentile(h_pos, 95))
        feat["h_skew"] = float(pd.Series(h_pos).skew()) if len(h_pos) > 2 else 0
        feat["h_kurtosis"] = float(pd.Series(h_pos).kurtosis()) if len(h_pos) > 3 else 0
        feat["h_cv"] = feat["h_std"] / feat["h_mean"] if feat["h_mean"] > 0 else 0
        feat["h_iqr"] = feat["h_p75"] - feat["h_p25"]

        if feat["h_max"] > 0:
            q1, q2, q3 = feat["h_max"] * 0.25, feat["h_max"] * 0.5, feat["h_max"] * 0.75
            feat["pct_lower_quarter"] = float(np.mean(h_pos < q1))
            feat["pct_mid_lower"] = float(np.mean((h_pos >= q1) & (h_pos < q2)))
            feat["pct_mid_upper"] = float(np.mean((h_pos >= q2) & (h_pos < q3)))
            feat["pct_upper_quarter"] = float(np.mean(h_pos >= q3))
        else:
            feat["pct_lower_quarter"] = feat["pct_mid_lower"] = feat["pct_mid_upper"] = feat["pct_upper_quarter"] = 0

        # RGB
        if has_rgb:
            cr, cg, cb = vr[idx], vg[idx], vb[idx]
            feat["r_mean"] = float(np.mean(cr))
            feat["g_mean"] = float(np.mean(cg))
            feat["b_mean"] = float(np.mean(cb))
            feat["r_std"] = float(np.std(cr))
            feat["g_std"] = float(np.std(cg))
            feat["b_std"] = float(np.std(cb))
            rgb_sum = feat["r_mean"] + feat["g_mean"] + feat["b_mean"]
            if rgb_sum > 0:
                feat["grvi"] = (feat["g_mean"] - feat["r_mean"]) / rgb_sum
                feat["ngrdi"] = (feat["g_mean"] - feat["r_mean"]) / (feat["g_mean"] + feat["r_mean"]) if (feat["g_mean"] + feat["r_mean"]) > 0 else 0
                feat["g_ratio"] = feat["g_mean"] / rgb_sum
                feat["r_ratio"] = feat["r_mean"] / rgb_sum
                feat["b_ratio"] = feat["b_mean"] / rgb_sum
            else:
                feat["grvi"] = feat["ngrdi"] = feat["g_ratio"] = feat["r_ratio"] = feat["b_ratio"] = 0
            feat["brightness"] = rgb_sum / 3.0
            feat["rgb_range"] = float(max(feat["r_mean"], feat["g_mean"], feat["b_mean"]) -
                                      min(feat["r_mean"], feat["g_mean"], feat["b_mean"]))

        # Intensity
        if has_intensity:
            ci = v_intensity[idx]
            feat["int_mean"] = float(np.mean(ci))
            feat["int_std"] = float(np.std(ci))
            feat["int_cv"] = feat["int_std"] / feat["int_mean"] if feat["int_mean"] > 0 else 0

        # Returns
        if has_returns:
            crn, cnr = v_return_num[idx], v_num_returns[idx]
            feat["pct_first_return"] = float(np.mean(crn == 1))
            feat["pct_single_return"] = float(np.mean(cnr == 1))
            feat["pct_multi_return"] = float(np.mean(cnr > 1))
            feat["mean_num_returns"] = float(np.mean(cnr))
            feat["penetration_ratio"] = 1.0 - feat["pct_first_return"]

        # Classification
        cc = v_class[idx]
        feat["pct_class3"] = float(np.mean(cc == 3))
        feat["pct_class4"] = float(np.mean(cc == 4))
        feat["pct_class5"] = float(np.mean(cc == 5))

        results.append(feat)

    return results


def main():
    log.info("=" * 60)
    log.info("Feature Extraction v2 (memory-safe, resume)")
    log.info("=" * 60)

    # Load trees grouped by tile
    log.info("Loading crown data...")
    df = pd.read_csv(TREES_CSV)
    tile_groups = defaultdict(list)
    for _, row in df.iterrows():
        tile_groups[row["tile"]].append({
            "tree_id": row["tree_id"],
            "centroid_x": row["centroid_x"],
            "centroid_y": row["centroid_y"],
            "crown_diameter_m": row["crown_diameter_m"],
        })
    del df
    gc.collect()

    # Find already-done tiles
    done_tiles = set()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV, usecols=["tile"])
        done_tiles = set(existing["tile"].unique())
        del existing
        gc.collect()
        log.info(f"Resuming: {len(done_tiles)} tiles already done")

    remaining = [(t, recs) for t, recs in tile_groups.items() if t not in done_tiles]
    log.info(f"Remaining: {len(remaining)} tiles")
    del tile_groups
    gc.collect()

    if not remaining:
        log.info("All tiles done!")
        return

    # Open CSV for append
    write_header = not os.path.exists(OUTPUT_CSV) or len(done_tiles) == 0
    csvfile = open(OUTPUT_CSV, "a", newline="")
    writer = None

    t_start = time.time()
    total_crowns = 0
    processed = 0

    # Process in batches with explicit GC
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_single_tile, tile, recs): tile
                for tile, recs in batch
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        if writer is None:
                            writer = csv.DictWriter(csvfile, fieldnames=list(result[0].keys()))
                            if write_header:
                                writer.writeheader()
                                write_header = False
                        for row in result:
                            writer.writerow(row)
                        total_crowns += len(result)
                    processed += 1
                except Exception as e:
                    processed += 1

        csvfile.flush()
        gc.collect()

        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        left = (len(remaining) - processed) / rate if rate > 0 else 0
        log.info(
            f"Batch done: {processed}/{len(remaining)} tiles, "
            f"{total_crowns:,} crowns [{elapsed:.0f}s, ~{left:.0f}s left]"
        )

    csvfile.close()
    elapsed = time.time() - t_start
    log.info(f"Complete: {total_crowns:,} crowns from {processed} tiles in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
