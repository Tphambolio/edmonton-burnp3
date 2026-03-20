#!/usr/bin/env python3
"""
Edmonton FBP Fuel Type Raster for BurnP3
==========================================
Generates fuel type and percent conifer rasters for natural/naturalized areas
following the Canadian Forest Fire Behaviour Prediction (FBP) System.

FBP Fuel Types used:
  C-2  (2)  — Boreal Spruce: >75% conifer, spruce-dominated
  C-3  (3)  — Mature Pine: >75% conifer, pine-dominated
  D-1  (11) — Leafless Aspen (placeholder — use D-2 for leaf-on)
  D-2  (12) — Green Aspen: >75% deciduous, poplar/aspen-dominated
  M-1  (13) — Boreal Mixedwood Leafless (placeholder)
  M-2  (14) — Boreal Mixedwood Green: 25-75% conifer
  O-1a (31) — Matted Grass
  O-1b (32) — Standing Grass (default for natural areas)
  NF   (99) — Non-fuel (urban, managed turf)
  WA   (98) — Water

Inputs:
  - Classified trees (conifer/deciduous, species)
  - Canopy tile results (canopy presence per 1m cell)
  - UPLVI natural area polygons
  - Naturalized area polygons
  - Edmonton boundary

Output:
  - fuel_type.tif — FBP fuel type integer codes (20m cells)
  - percent_conifer.tif — conifer fraction 0-100 for M-1/M-2 (20m cells)
"""

import os
import json
import logging
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import shape, box, mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fuel")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"
TREES_V2_CSV = os.path.join(BASE_DIR, "tree_results/trees_classified_v2.csv")
TREES_V1_CSV = os.path.join(BASE_DIR, "tree_results/individual_trees.csv")
CANOPY_CSV = os.path.join(BASE_DIR, "canopy_tile_results.csv")
UPLVI_JSON = "/tmp/edmonton_uplvi.json"
NAT_JSON = "/tmp/edmonton_naturalized.json"
BOUNDARY_FILE = "/tmp/edmonton_boundary_3776.geojson"

OUTPUT_DIR = os.path.join(BASE_DIR, "fuel_raster")
FUEL_TIF = os.path.join(OUTPUT_DIR, "fuel_type.tif")
PC_TIF = os.path.join(OUTPUT_DIR, "percent_conifer.tif")
REPORT_FILE = os.path.join(OUTPUT_DIR, "fuel_classification_report.txt")

CELL_SIZE = 20  # metres — standard BurnP3 resolution
CRS = "EPSG:3776"

# FBP fuel type codes
FBP = {
    "C-2": 2,    # Boreal Spruce
    "C-3": 3,    # Mature Jack/Lodgepole Pine
    "D-2": 12,   # Green Aspen
    "M-2": 14,   # Boreal Mixedwood Green
    "O-1b": 32,  # Standing Grass
    "NF": 99,    # Non-fuel
    "WA": 98,    # Water
}

# Thresholds
CONIFER_DOMINANT = 0.75   # >75% conifer → C-type
DECIDUOUS_DOMINANT = 0.75 # >75% deciduous → D-type
MIN_CANOPY_COVER = 0.10   # <10% canopy = grass or non-fuel
MIN_TREES_PER_CELL = 2    # minimum trees to assign forest fuel type


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_trees():
    """Load classified trees with species info."""
    log.info("Loading classified trees...")

    cols_v2 = ["tile", "tree_id", "centroid_x", "centroid_y", "max_height",
               "crown_area_m2", "veg_type", "leaf_type_v2"]
    cols_v1_base = ["tile", "tree_id", "centroid_x", "centroid_y", "max_height",
                    "crown_area_m2", "veg_type"]

    if os.path.exists(TREES_V2_CSV):
        v2 = pd.read_csv(TREES_V2_CSV, usecols=cols_v2)
        v2 = v2.rename(columns={"leaf_type_v2": "leaf_type"})
        log.info(f"  v2: {len(v2):,} trees")

        # Fill in remainder from v1 (no leaf_type column)
        v1 = pd.read_csv(TREES_V1_CSV, usecols=cols_v1_base)
        v1["leaf_type"] = "deciduous"  # conservative default
        v1_only = v1[~v1.set_index(["tile", "tree_id"]).index.isin(
            v2.set_index(["tile", "tree_id"]).index
        )]
        log.info(f"  v1 remainder: {len(v1_only):,} (defaulted to deciduous)")

        df = pd.concat([v2, v1_only], ignore_index=True)
    else:
        df = pd.read_csv(TREES_V1_CSV, usecols=cols_v1_base)
        df["leaf_type"] = "deciduous"

    log.info(f"  Total: {len(df):,} trees")
    return df


def load_natural_areas():
    """Load and merge UPLVI vegetated + naturalized area polygons."""
    log.info("Loading natural area polygons...")

    # UPLVI vegetated areas
    with open(UPLVI_JSON) as f:
        uplvi_raw = json.load(f)

    uplvi_feats = []
    for r in uplvi_raw:
        if r.get("primeclas1") != "Vegetated":
            continue
        geom = r.get("the_geom")
        if not geom:
            continue
        try:
            uplvi_feats.append({"geometry": shape(geom), "source": "UPLVI", "type": "Vegetated"})
        except Exception:
            continue

    uplvi_gdf = gpd.GeoDataFrame(uplvi_feats, crs="EPSG:4326").to_crs(CRS)
    log.info(f"  UPLVI vegetated: {len(uplvi_gdf)} polygons")

    # Naturalized areas
    with open(NAT_JSON) as f:
        nat_raw = json.load(f)

    nat_feats = []
    for r in nat_raw:
        geom = r.get("geometry_multipolygon")
        if not geom:
            continue
        try:
            nat_feats.append({
                "geometry": shape(geom),
                "source": "Naturalized",
                "type": r.get("vegetation_type", "Unknown"),
            })
        except Exception:
            continue

    nat_gdf = gpd.GeoDataFrame(nat_feats, crs="EPSG:4326").to_crs(CRS)
    log.info(f"  Naturalized: {len(nat_gdf)} polygons")

    # Merge
    combined = pd.concat([uplvi_gdf, nat_gdf], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs=CRS)

    # Dissolve to single multipolygon for rasterization
    natural_union = combined.dissolve().geometry.iloc[0]
    log.info(f"  Combined natural area: {natural_union.area / 1e6:.1f} km²")

    return combined, natural_union


def load_boundary():
    gdf = gpd.read_file(BOUNDARY_FILE)
    if gdf.crs != CRS:
        gdf = gdf.to_crs(CRS)
    return gdf.geometry.iloc[0]


# ---------------------------------------------------------------------------
# Build fuel raster
# ---------------------------------------------------------------------------
def build_fuel_rasters(trees_df, natural_gdf, natural_union, boundary):
    """Build fuel type and percent conifer rasters."""
    log.info("Building fuel rasters...")

    # Raster extent from boundary
    bnd = boundary.bounds  # (minx, miny, maxx, maxy)
    xmin = int(np.floor(bnd[0] / CELL_SIZE) * CELL_SIZE)
    ymin = int(np.floor(bnd[1] / CELL_SIZE) * CELL_SIZE)
    xmax = int(np.ceil(bnd[2] / CELL_SIZE) * CELL_SIZE)
    ymax = int(np.ceil(bnd[3] / CELL_SIZE) * CELL_SIZE)

    nx = int((xmax - xmin) / CELL_SIZE)
    ny = int((ymax - ymin) / CELL_SIZE)
    log.info(f"  Raster: {nx}x{ny} cells at {CELL_SIZE}m ({(xmax-xmin)/1000:.0f} x {(ymax-ymin)/1000:.0f} km)")

    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)

    # Initialize rasters
    fuel = np.full((ny, nx), FBP["NF"], dtype=np.int16)
    pc = np.zeros((ny, nx), dtype=np.int16)

    # Step 1: Rasterize natural area mask
    log.info("  Rasterizing natural area mask...")
    natural_mask = rasterize(
        [(mapping(natural_union), 1)],
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    natural_cells = natural_mask.sum()
    log.info(f"  Natural area cells: {natural_cells:,} ({100*natural_cells/(nx*ny):.1f}%)")

    # Step 2: Assign trees to grid cells
    log.info("  Assigning trees to grid cells...")
    tree_col = ((trees_df["centroid_x"].values - xmin) / CELL_SIZE).astype(int)
    tree_row = ny - 1 - ((trees_df["centroid_y"].values - ymin) / CELL_SIZE).astype(int)

    # Filter to valid cells
    valid = (tree_col >= 0) & (tree_col < nx) & (tree_row >= 0) & (tree_row < ny)
    tree_col = tree_col[valid]
    tree_row = tree_row[valid]
    leaf_types = trees_df["leaf_type"].values[valid]
    veg_types = trees_df["veg_type"].values[valid]
    heights = trees_df["max_height"].values[valid]

    log.info(f"  {valid.sum():,} trees in raster extent")

    # Step 3: Aggregate per cell — count conifers and deciduous
    log.info("  Computing per-cell conifer fractions...")
    conifer_count = np.zeros((ny, nx), dtype=np.int32)
    deciduous_count = np.zeros((ny, nx), dtype=np.int32)
    tree_count = np.zeros((ny, nx), dtype=np.int32)
    max_height_grid = np.zeros((ny, nx), dtype=np.float32)

    is_conifer = leaf_types == "conifer"
    is_deciduous = leaf_types == "deciduous"
    is_tree = veg_types == "tree"  # >5m, not shrub

    # Use np.add.at for unbuffered accumulation
    np.add.at(conifer_count, (tree_row[is_conifer], tree_col[is_conifer]), 1)
    np.add.at(deciduous_count, (tree_row[is_deciduous], tree_col[is_deciduous]), 1)
    np.add.at(tree_count, (tree_row, tree_col), 1)
    np.maximum.at(max_height_grid, (tree_row, tree_col), heights)

    total_trees = conifer_count + deciduous_count
    conifer_frac = np.where(total_trees > 0, conifer_count / total_trees, 0).astype(np.float32)

    # Step 4: Classify fuel types within natural areas
    log.info("  Classifying fuel types...")

    # Natural areas with enough trees → forest fuel types
    forested = natural_mask & (tree_count >= MIN_TREES_PER_CELL)

    # Conifer-dominant
    c_dominant = forested & (conifer_frac >= CONIFER_DOMINANT)
    # For Edmonton: spruce is the dominant conifer, so C-2
    fuel[c_dominant] = FBP["C-2"]

    # Deciduous-dominant
    d_dominant = forested & (conifer_frac <= (1 - DECIDUOUS_DOMINANT))
    fuel[d_dominant] = FBP["D-2"]

    # Mixedwood (neither fully conifer nor fully deciduous)
    mixedwood = forested & ~c_dominant & ~d_dominant
    fuel[mixedwood] = FBP["M-2"]

    # Natural areas with few/no trees → grass
    grass = natural_mask & (tree_count < MIN_TREES_PER_CELL)
    fuel[grass] = FBP["O-1b"]

    # Percent conifer (0-100) for M-2 cells
    pc = (conifer_frac * 100).astype(np.int16)

    # Step 5: Mask to boundary
    log.info("  Masking to city boundary...")
    boundary_mask = rasterize(
        [(mapping(boundary), 1)],
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    fuel[~boundary_mask] = 0  # nodata outside boundary

    # Stats
    in_boundary = boundary_mask.sum()
    fuel_counts = {}
    for name, code in FBP.items():
        count = ((fuel == code) & boundary_mask).sum()
        if count > 0:
            fuel_counts[name] = int(count)

    log.info("  Fuel type distribution:")
    for name, count in sorted(fuel_counts.items(), key=lambda x: -x[1]):
        area_ha = count * CELL_SIZE * CELL_SIZE / 1e4
        pct = 100 * count / in_boundary
        log.info(f"    {name:>5s}: {count:>8,} cells  ({area_ha:>8,.0f} ha, {pct:>5.1f}%)")

    return fuel, pc, transform, nx, ny, fuel_counts, in_boundary


# ---------------------------------------------------------------------------
# Write rasters
# ---------------------------------------------------------------------------
def write_raster(data, path, transform, nx, ny, nodata=0, dtype="int16"):
    """Write a single-band GeoTIFF."""
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": nx,
        "height": ny,
        "count": 1,
        "crs": CRS,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)
    log.info(f"  Written: {path} ({os.path.getsize(path)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Edmonton FBP Fuel Type Raster for BurnP3")
    log.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trees_df = load_trees()
    natural_gdf, natural_union = load_natural_areas()
    boundary = load_boundary()

    fuel, pc, transform, nx, ny, fuel_counts, total_cells = build_fuel_rasters(
        trees_df, natural_gdf, natural_union, boundary
    )

    log.info("Writing rasters...")
    write_raster(fuel, FUEL_TIF, transform, nx, ny, nodata=0)
    write_raster(pc, PC_TIF, transform, nx, ny, nodata=-1)

    # Generate report
    cell_area_ha = CELL_SIZE * CELL_SIZE / 1e4

    report = f"""
================================================================================
  EDMONTON FBP FUEL TYPE CLASSIFICATION — BurnP3 Input
================================================================================

  Source Data:     2025 City of Edmonton Vegetation LiDAR
  Classifier:     Conifer/deciduous RF v2 (90.4% accuracy)
  Natural Areas:  UPLVI vegetated + Naturalized Areas (Edmonton Open Data)
  Cell Size:       {CELL_SIZE}m x {CELL_SIZE}m
  CRS:             EPSG:3776 (NAD83 / Alberta 3TM ref merid 114 W)
  Grid Size:       {nx} x {ny} cells

  OUTPUT FILES
  ------------
  Fuel type raster:     {FUEL_TIF}
  Percent conifer:      {PC_TIF}

  FUEL TYPE DISTRIBUTION (within Edmonton boundary)
  --------------------------------------------------
"""
    for name, code in sorted(FBP.items(), key=lambda x: x[1]):
        count = fuel_counts.get(name, 0)
        area_ha = count * cell_area_ha
        pct = 100 * count / total_cells if total_cells > 0 else 0
        report += f"  {name:>5s} ({code:>2d}):  {count:>8,} cells  {area_ha:>8,.0f} ha  {pct:>5.1f}%\n"

    total_fuel = sum(v for k, v in fuel_counts.items() if k != "NF")
    report += f"""
  Total fuel cells:     {total_fuel:,} ({total_fuel * cell_area_ha:,.0f} ha)
  Non-fuel (NF) cells:  {fuel_counts.get('NF', 0):,} ({fuel_counts.get('NF', 0) * cell_area_ha:,.0f} ha)

  CLASSIFICATION RULES
  --------------------
  Within UPLVI Vegetated + Naturalized Area polygons:
    >={int(CONIFER_DOMINANT*100)}% conifer trees      → C-2 (Boreal Spruce)
    >={int(DECIDUOUS_DOMINANT*100)}% deciduous trees   → D-2 (Green Aspen)
    25-75% conifer            → M-2 (Boreal Mixedwood Green) + PC%
    <{MIN_TREES_PER_CELL} trees per cell       → O-1b (Standing Grass)
  Outside natural areas:
    All cells                 → NF (Non-fuel)

  NOTES
  -----
  - Leaf-on collection: D-2 and M-2 (green variants) used throughout
  - For leaf-off modeling, substitute D-1 for D-2 and M-1 for M-2
  - Percent conifer raster provided for M-2 cells (BurnP3 PC input)
  - Spruce is Edmonton's dominant conifer; C-2 used over C-3
  - Grass classification based on UPLVI/naturalized boundaries, not spectral
  - Urban canopy trees outside natural areas classified as NF (not fuel)

================================================================================
"""
    print(report)
    with open(REPORT_FILE, "w") as f:
        f.write(report)

    elapsed = time.time() - t0
    log.info(f"Complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
