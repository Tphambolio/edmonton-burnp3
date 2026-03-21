#!/usr/bin/env python3
"""
Elm Genus Detector with Orthophoto Features
=============================================
Extracts 7.5cm RGB ortho texture features per crown and combines
with LiDAR structural features for binary Elm detection.
"""

import os
import json
import logging
import time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from pyproj import Transformer
import lightgbm as lgb
import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("elm")

BASE = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"
ORTHO_DIR = os.path.join(BASE, "ortho_tiles")
FEATURES_CSV = os.path.join(BASE, "tree_results/crown_features.csv")
TREES_CSV = os.path.join(BASE, "tree_results/trees_classified.csv")
INVENTORY = os.path.join(BASE, "edmonton_tree_inventory.json")


def build_ortho_index():
    """Build spatial index of ortho tile extents."""
    log.info("Building ortho tile index...")
    tiles = []
    for f in sorted(glob.glob(os.path.join(ORTHO_DIR, "*.tif"))):
        with rasterio.open(f) as src:
            b = src.bounds
            tiles.append({
                "path": f,
                "xmin": b.left, "ymin": b.bottom,
                "xmax": b.right, "ymax": b.top,
                "res": src.res[0],
            })
    log.info(f"  {len(tiles)} ortho tiles indexed")
    return tiles


def find_ortho_tile(x, y, ortho_index):
    """Find which ortho tile contains a point."""
    for t in ortho_index:
        if t["xmin"] <= x <= t["xmax"] and t["ymin"] <= y <= t["ymax"]:
            return t
    return None


def extract_ortho_features(x, y, radius, tile_info):
    """Extract RGB features from ortho within crown radius."""
    try:
        with rasterio.open(tile_info["path"]) as src:
            # Window around the crown
            buf = radius + 1
            window = from_bounds(x - buf, y - buf, x + buf, y + buf, src.transform)

            # Read RGB bands
            data = src.read(window=window)  # (3, h, w)
            if data.size == 0:
                return None

            r, g, b = data[0].astype(np.float32), data[1].astype(np.float32), data[2].astype(np.float32)

            # Create circular mask
            h, w = r.shape
            cy, cx = h / 2, w / 2
            yy, xx = np.ogrid[:h, :w]
            pix_radius = radius / tile_info["res"]
            mask = ((xx - cx)**2 + (yy - cy)**2) <= pix_radius**2

            if mask.sum() < 10:
                return None

            rm, gm, bm = r[mask], g[mask], b[mask]

            eps = 1e-6
            rgb_sum = rm.mean() + gm.mean() + bm.mean() + eps

            feats = {
                "ortho_r_mean": float(rm.mean()),
                "ortho_g_mean": float(gm.mean()),
                "ortho_b_mean": float(bm.mean()),
                "ortho_r_std": float(rm.std()),
                "ortho_g_std": float(gm.std()),
                "ortho_b_std": float(bm.std()),
                "ortho_brightness": float(rgb_sum / 3),
                "ortho_brightness_std": float(np.std([rm.mean(), gm.mean(), bm.mean()])),
                "ortho_r_chrom": float(rm.mean() / rgb_sum),
                "ortho_g_chrom": float(gm.mean() / rgb_sum),
                "ortho_b_chrom": float(bm.mean() / rgb_sum),
                "ortho_ndgr": float((gm.mean() - rm.mean()) / (gm.mean() + rm.mean() + eps)),
                "ortho_excess_green": float(2 * gm.mean() - rm.mean() - bm.mean()),
                "ortho_green_dom": float((gm > rm).mean()),  # fraction of pixels where green > red
                "ortho_shadow_frac": float((rm + gm + bm < 100).mean()),  # dark pixel fraction
                # Texture features
                "ortho_r_range": float(np.percentile(rm, 95) - np.percentile(rm, 5)),
                "ortho_g_range": float(np.percentile(gm, 95) - np.percentile(gm, 5)),
                "ortho_b_range": float(np.percentile(bm, 95) - np.percentile(bm, 5)),
                "ortho_r_cv": float(rm.std() / (rm.mean() + eps)),
                "ortho_g_cv": float(gm.std() / (gm.mean() + eps)),
                "ortho_b_cv": float(bm.std() / (bm.mean() + eps)),
                # Edge/texture proxy
                "ortho_r_entropy": float(np.std(np.diff(rm.ravel()[:100]))),
                "ortho_g_entropy": float(np.std(np.diff(gm.ravel()[:100]))),
            }
            return feats
    except Exception:
        return None


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Elm Genus Detector with Ortho Features")
    log.info("=" * 60)

    # Load LiDAR features
    log.info("Loading crown features...")
    feat = pd.read_csv(FEATURES_CSV)
    trees = pd.read_csv(TREES_CSV, usecols=[
        'tile', 'tree_id', 'centroid_x', 'centroid_y', 'max_height',
        'crown_area_m2', 'crown_diameter_m', 'compactness', 'eccentricity',
        'veg_type', 'perimeter_m', 'mean_height', 'std_height'])
    merged = trees.merge(feat, on=['tile', 'tree_id'], how='inner')
    log.info(f"Crowns with LiDAR features: {len(merged):,}")

    # Load inventory → binary Elm labels
    log.info("Loading inventory...")
    with open(INVENTORY) as f:
        inv_raw = json.load(f)

    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3776', always_xy=True)
    inv = []
    for t in inv_raw:
        genus = t.get('genus', '')
        lat, lon = t.get('latitude'), t.get('longitude')
        if not lat or not lon or not genus:
            continue
        x, y = transformer.transform(float(lon), float(lat))
        inv.append({'x': x, 'y': y, 'is_elm': 1 if genus == 'Ulmus' else 0})

    log.info(f"Inventory: {len(inv):,} ({sum(t['is_elm'] for t in inv):,} elms)")

    # Spatial match
    log.info("Spatial matching...")
    crown_xy = merged[['centroid_x', 'centroid_y']].values
    kdtree = cKDTree(crown_xy)
    inv_xy = np.array([[t['x'], t['y']] for t in inv])
    dists, idxs = kdtree.query(inv_xy, k=1)

    labels = {}
    for i, (d, idx) in enumerate(zip(dists, idxs)):
        if d <= 5.0 and idx not in labels:
            labels[idx] = inv[i]['is_elm']

    log.info(f"Matched: {len(labels):,} ({sum(labels.values()):,} elms)")

    # Build ortho index
    ortho_index = build_ortho_index()

    # Extract ortho features for matched crowns
    log.info("Extracting ortho features for training crowns...")
    train_idx = list(labels.keys())
    train_df = merged.iloc[train_idx].copy()
    train_df['is_elm'] = [labels[i] for i in train_idx]

    ortho_feats = []
    hits = 0
    for i, (_, row) in enumerate(train_df.iterrows()):
        tile = find_ortho_tile(row['centroid_x'], row['centroid_y'], ortho_index)
        if tile:
            of = extract_ortho_features(
                row['centroid_x'], row['centroid_y'],
                row['crown_diameter_m'] / 2, tile
            )
            if of:
                ortho_feats.append(of)
                hits += 1
            else:
                ortho_feats.append({})
        else:
            ortho_feats.append({})

        if (i + 1) % 50000 == 0:
            log.info(f"  {i+1:,}/{len(train_df):,} crowns ({hits:,} with ortho)")

    log.info(f"Ortho coverage: {hits:,}/{len(train_df):,} ({100*hits/len(train_df):.1f}%)")

    # Merge ortho features
    ortho_df = pd.DataFrame(ortho_feats)
    train_df = pd.concat([train_df.reset_index(drop=True), ortho_df.reset_index(drop=True)], axis=1)

    # Fill NaN ortho features with 0 (trees without ortho coverage)
    train_df = train_df.fillna(0)
    train_df['has_ortho'] = (ortho_df.sum(axis=1) > 0).astype(int)

    # Select features
    exclude = {'tile', 'tree_id', 'centroid_x', 'centroid_y', 'veg_type', 'n_points', 'is_elm'}
    feature_cols = [c for c in train_df.columns if c not in exclude]
    log.info(f"Features: {len(feature_cols)}")

    X = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = train_df['is_elm'].values

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    log.info(f"Train: {len(X_train):,} ({y_train.sum():,} elm)")
    log.info(f"Val: {len(X_val):,} ({y_val.sum():,} elm)")

    # === Model A: LiDAR only (baseline) ===
    lidar_cols = [i for i, c in enumerate(feature_cols) if not c.startswith('ortho_') and c != 'has_ortho']
    clf_lidar = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=-1, num_leaves=127,
        min_child_samples=10, subsample=0.9, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    clf_lidar.fit(X_train[:, lidar_cols], y_train)
    p_lidar = clf_lidar.predict(X_val[:, lidar_cols])
    f1_lidar = f1_score(y_val, p_lidar, average='macro')

    log.info(f"\n=== LiDAR ONLY — Macro F1: {f1_lidar:.4f} ===")
    print(classification_report(y_val, p_lidar, target_names=['Non-Elm', 'Elm'], digits=3))

    # === Model B: LiDAR + Ortho ===
    clf_full = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=-1, num_leaves=127,
        min_child_samples=10, subsample=0.9, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    clf_full.fit(X_train, y_train)
    p_full = clf_full.predict(X_val)
    f1_full = f1_score(y_val, p_full, average='macro')

    log.info(f"\n=== LiDAR + ORTHO — Macro F1: {f1_full:.4f} ===")
    print(classification_report(y_val, p_full, target_names=['Non-Elm', 'Elm'], digits=3))

    # Cross-val
    cv = cross_val_score(clf_full, X, y, cv=5, scoring='f1_macro')
    log.info(f"5-fold CV: {cv.mean():.4f} ± {cv.std():.4f}")

    # Feature importance — top 15
    importances = sorted(zip(feature_cols, clf_full.feature_importances_), key=lambda x: -x[1])
    log.info("\nTop 15 features:")
    for name, imp in importances[:15]:
        marker = " ** ORTHO" if name.startswith("ortho_") else ""
        log.info(f"  {name:<30s} {imp:>5.0f}{marker}")

    log.info(f"\nImprovement: LiDAR {f1_lidar:.4f} → LiDAR+Ortho {f1_full:.4f} ({(f1_full-f1_lidar)*100:+.2f} pp)")
    log.info(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
