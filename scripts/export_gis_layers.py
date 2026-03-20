#!/usr/bin/env python3
"""
Export Edmonton Canopy Analysis to GIS Layers (GeoPackage)
===========================================================
Generates spatial layers from all analysis products:
1. Canopy cover heatmap (per-tile polygons)
2. Individual tree points (5.17M with species classification)
3. Canopy by UPLVI ecoclass
4. Canopy by naturalized area
5. Edmonton boundary with canopy stats
"""

import os
import logging
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gis")

BASE_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"
OUTPUT_GPKG = os.path.join(BASE_DIR, "edmonton_canopy_analysis.gpkg")

TILE_CSV = os.path.join(BASE_DIR, "canopy_tile_results.csv")
TREES_V1_CSV = os.path.join(BASE_DIR, "tree_results/individual_trees.csv")
TREES_V2_CSV = os.path.join(BASE_DIR, "tree_results/trees_classified_v2.csv")
FEATURES_CSV = os.path.join(BASE_DIR, "tree_results/crown_features.csv")
BOUNDARY_FILE = "/tmp/edmonton_boundary_3776.geojson"


def export_canopy_tiles():
    """Layer 1: Per-tile canopy cover as polygon grid with attributes."""
    log.info("Building canopy tile grid...")
    df = pd.read_csv(TILE_CSV)

    geometries = []
    for _, row in df.iterrows():
        geometries.append(box(row["xmin"], row["ymin"], row["xmax"], row["ymax"]))

    df["canopy_pct"] = 100 * df["canopy_cells"] / df["boundary_cells"].clip(lower=1)
    df["canopy_area_ha"] = df["canopy_cells"] / 1e4
    df["low_pct"] = 100 * df["low_cells"] / df["boundary_cells"].clip(lower=1)
    df["med_pct"] = 100 * df["med_cells"] / df["boundary_cells"].clip(lower=1)
    df["high_pct"] = 100 * df["high_cells"] / df["boundary_cells"].clip(lower=1)

    gdf = gpd.GeoDataFrame(
        df[["tile", "canopy_pct", "canopy_area_ha", "low_pct", "med_pct", "high_pct",
            "boundary_cells", "canopy_cells"]],
        geometry=geometries,
        crs="EPSG:3776",
    )

    gdf.to_file(OUTPUT_GPKG, layer="canopy_tiles", driver="GPKG")
    log.info(f"  canopy_tiles: {len(gdf)} polygons")
    return gdf


def export_tree_points():
    """Layer 2: Individual trees as points with classification."""
    log.info("Building tree point layer...")

    # Use v2 classified trees where available, fall back to v1
    if os.path.exists(TREES_V2_CSV):
        log.info("  Loading v2 classified trees (with features)...")
        v2 = pd.read_csv(TREES_V2_CSV, usecols=[
            "tile", "tree_id", "top_x", "top_y", "centroid_x", "centroid_y",
            "max_height", "mean_height", "crown_area_m2", "crown_diameter_m",
            "compactness", "eccentricity", "veg_type", "leaf_type_v2", "conifer_prob_v2",
        ])
        v2 = v2.rename(columns={"leaf_type_v2": "leaf_type", "conifer_prob_v2": "conifer_prob"})
        log.info(f"  v2: {len(v2):,} trees")

        # Load v1 for trees not in v2
        v1 = pd.read_csv(TREES_V1_CSV)
        v1_only = v1[~v1.set_index(["tile", "tree_id"]).index.isin(
            v2.set_index(["tile", "tree_id"]).index
        )]
        log.info(f"  v1 remainder: {len(v1_only):,} trees")

        # Align columns
        for col in ["leaf_type", "conifer_prob"]:
            if col not in v1_only.columns:
                v1_only[col] = "unknown" if col == "leaf_type" else 0.5

        common_cols = ["tile", "tree_id", "top_x", "top_y", "centroid_x", "centroid_y",
                       "max_height", "mean_height", "crown_area_m2", "crown_diameter_m",
                       "compactness", "eccentricity", "veg_type", "leaf_type", "conifer_prob"]

        v2_sub = v2[[c for c in common_cols if c in v2.columns]]
        v1_sub = v1_only[[c for c in common_cols if c in v1_only.columns]]

        df = pd.concat([v2_sub, v1_sub], ignore_index=True)
    else:
        log.info("  Loading v1 trees only...")
        df = pd.read_csv(TREES_V1_CSV)
        df["conifer_prob"] = 0.5

    log.info(f"  Total: {len(df):,} trees")

    # Sample for GIS layer — 5M points is too many for a GPKG
    # Export full dataset as CSV, subsample for spatial layer
    SAMPLE_SIZE = 500_000
    if len(df) > SAMPLE_SIZE:
        log.info(f"  Sampling {SAMPLE_SIZE:,} trees for spatial layer (full data in CSV)")
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        df_sample = df

    geometry = [Point(x, y) for x, y in zip(df_sample["centroid_x"], df_sample["centroid_y"])]

    gdf = gpd.GeoDataFrame(df_sample, geometry=geometry, crs="EPSG:3776")

    # Drop coordinate columns (redundant with geometry)
    drop_cols = [c for c in ["centroid_x", "centroid_y", "top_x", "top_y"] if c in gdf.columns]
    gdf = gdf.drop(columns=drop_cols)

    gdf.to_file(OUTPUT_GPKG, layer="trees_sample_500k", driver="GPKG")
    log.info(f"  trees_sample_500k: {len(gdf)} points")

    # Also export a trees-by-tile summary as polygons
    return gdf


def export_tree_density_grid():
    """Layer 3: Tree density and species composition per tile."""
    log.info("Building tree density grid...")

    v1 = pd.read_csv(TREES_V1_CSV, usecols=["tile", "tree_id", "max_height", "crown_area_m2", "veg_type"])
    tiles = pd.read_csv(TILE_CSV, usecols=["tile", "xmin", "ymin", "xmax", "ymax", "boundary_cells"])

    # Aggregate per tile
    agg = v1.groupby("tile").agg(
        tree_count=("tree_id", "count"),
        mean_height=("max_height", "mean"),
        total_crown_ha=("crown_area_m2", lambda x: x.sum() / 1e4),
        tree_pct=("veg_type", lambda x: (x == "tree").mean() * 100),
    ).reset_index()

    merged = tiles.merge(agg, on="tile", how="left").fillna(0)
    merged["trees_per_ha"] = merged["tree_count"] / (merged["boundary_cells"].clip(lower=1) / 1e4)

    geometries = [box(r["xmin"], r["ymin"], r["xmax"], r["ymax"]) for _, r in merged.iterrows()]

    gdf = gpd.GeoDataFrame(
        merged[["tile", "tree_count", "trees_per_ha", "mean_height", "total_crown_ha", "tree_pct"]],
        geometry=geometries,
        crs="EPSG:3776",
    )

    gdf.to_file(OUTPUT_GPKG, layer="tree_density_grid", driver="GPKG")
    log.info(f"  tree_density_grid: {len(gdf)} polygons")
    return gdf


def export_boundary():
    """Layer 4: Edmonton boundary with summary stats."""
    log.info("Building boundary layer...")

    gdf = gpd.read_file(BOUNDARY_FILE)
    if gdf.crs != "EPSG:3776":
        gdf = gdf.to_crs("EPSG:3776")

    gdf["canopy_cover_pct"] = 15.2
    gdf["canopy_area_km2"] = 119.6
    gdf["total_area_km2"] = 784.5
    gdf["trees_detected"] = 5168142
    gdf["classifier_accuracy"] = 90.4

    gdf.to_file(OUTPUT_GPKG, layer="edmonton_boundary", driver="GPKG")
    log.info(f"  edmonton_boundary: {len(gdf)} polygon(s)")
    return gdf


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Exporting GIS Layers to GeoPackage")
    log.info(f"Output: {OUTPUT_GPKG}")
    log.info("=" * 60)

    # Remove existing GPKG to start fresh
    if os.path.exists(OUTPUT_GPKG):
        os.remove(OUTPUT_GPKG)

    export_boundary()
    export_canopy_tiles()
    export_tree_density_grid()
    export_tree_points()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(OUTPUT_GPKG) / 1e6

    log.info("=" * 60)
    log.info(f"Done in {elapsed:.0f}s")
    log.info(f"Output: {OUTPUT_GPKG} ({size_mb:.0f} MB)")
    log.info("Layers: edmonton_boundary, canopy_tiles, tree_density_grid, trees_sample_500k")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
