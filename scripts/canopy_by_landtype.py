#!/usr/bin/env python3
"""
Edmonton Canopy Cover by Land Type
====================================
Overlays classified tree crowns with UPLVI ecoclasses and Naturalized Areas
to compute canopy cover percentages by land type — matching Challenger's
deliverable scope.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape, MultiPolygon
from pyproj import Transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("overlay")

BASE_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"
TREES_CSV = os.path.join(BASE_DIR, "tree_results/trees_classified.csv")
CANOPY_CSV = os.path.join(BASE_DIR, "canopy_tile_results.csv")
UPLVI_JSON = "/tmp/edmonton_uplvi.json"
NAT_JSON = "/tmp/edmonton_naturalized.json"
BOUNDARY_FILE = "/tmp/edmonton_boundary_3776.geojson"
OUTPUT_DIR = os.path.join(BASE_DIR, "tree_results")
REPORT_FILE = os.path.join(OUTPUT_DIR, "canopy_by_landtype_report.txt")


def load_trees():
    """Load classified trees as GeoDataFrame."""
    log.info("Loading classified trees...")
    df = pd.read_csv(TREES_CSV)
    geometry = [Point(x, y) for x, y in zip(df["centroid_x"], df["centroid_y"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:3776")
    log.info(f"Loaded {len(gdf):,} trees")
    return gdf


def load_uplvi():
    """Load UPLVI polygons as GeoDataFrame in EPSG:3776."""
    log.info("Loading UPLVI...")
    with open(UPLVI_JSON) as f:
        data = json.load(f)

    features = []
    for r in data:
        geom_data = r.get("the_geom")
        if not geom_data:
            continue
        try:
            geom = shape(geom_data)
            features.append({
                "geometry": geom,
                "primary_class": r.get("primeclas1", "Unknown"),
                "ecounit": r.get("ecounit1", ""),
                "land_class": r.get("landclas1", ""),
                "stand_type": r.get("stnd1", ""),
                "moisture": r.get("moisture1", ""),
            })
        except Exception:
            continue

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3776")
    log.info(f"Loaded {len(gdf):,} UPLVI polygons")
    return gdf


def load_naturalized():
    """Load Naturalized Areas as GeoDataFrame in EPSG:3776."""
    log.info("Loading Naturalized Areas...")
    with open(NAT_JSON) as f:
        data = json.load(f)

    features = []
    for r in data:
        geom_data = r.get("geometry_multipolygon")
        if not geom_data:
            continue
        try:
            geom = shape(geom_data)
            features.append({
                "geometry": geom,
                "vegetation_type": r.get("vegetation_type", "Unknown"),
                "nat_type": r.get("type", ""),
                "owner": r.get("owner", ""),
                "area_m2": float(r.get("area", 0)),
            })
        except Exception:
            continue

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3776")
    log.info(f"Loaded {len(gdf):,} Naturalized Areas")
    return gdf


def load_boundary():
    """Load Edmonton boundary."""
    gdf = gpd.read_file(BOUNDARY_FILE)
    if gdf.crs != "EPSG:3776":
        gdf = gdf.to_crs("EPSG:3776")
    return gdf


def compute_canopy_by_overlay(trees_gdf, overlay_gdf, overlay_name, group_col):
    """
    Spatial join trees to overlay polygons and compute canopy stats by group.
    """
    log.info(f"Spatial join: trees → {overlay_name}...")

    # Spatial join
    joined = gpd.sjoin(trees_gdf, overlay_gdf, how="inner", predicate="within")
    log.info(f"  {len(joined):,} trees fall within {overlay_name} polygons")

    if len(joined) == 0:
        return pd.DataFrame()

    # Stats by group
    results = []
    for group_val, group_df in joined.groupby(group_col):
        total_trees = len(group_df)
        total_area = group_df["crown_area_m2"].sum()

        trees_only = group_df[group_df["veg_type"] == "tree"]
        shrubs_only = group_df[group_df["veg_type"] == "shrub"]

        conifer = group_df[group_df["leaf_type"] == "conifer"]
        deciduous = group_df[group_df["leaf_type"] == "deciduous"]

        tree_conifer = trees_only[trees_only["leaf_type"] == "conifer"]
        tree_deciduous = trees_only[trees_only["leaf_type"] == "deciduous"]
        shrub_conifer = shrubs_only[shrubs_only["leaf_type"] == "conifer"]
        shrub_deciduous = shrubs_only[shrubs_only["leaf_type"] == "deciduous"]

        results.append({
            "category": group_val,
            "total_crowns": total_trees,
            "total_canopy_area_ha": round(total_area / 1e4, 1),
            "tree_count": len(trees_only),
            "shrub_count": len(shrubs_only),
            "tree_deciduous": len(tree_deciduous),
            "tree_conifer": len(tree_conifer),
            "shrub_deciduous": len(shrub_deciduous),
            "shrub_conifer": len(shrub_conifer),
            "conifer_pct": round(100 * len(conifer) / total_trees, 1) if total_trees > 0 else 0,
        })

    return pd.DataFrame(results).sort_values("total_crowns", ascending=False)


def main():
    log.info("=" * 60)
    log.info("Edmonton Canopy Cover by Land Type")
    log.info("=" * 60)

    # Load data
    trees_gdf = load_trees()
    uplvi_gdf = load_uplvi()
    nat_gdf = load_naturalized()

    # Canopy by UPLVI primary class
    uplvi_results = compute_canopy_by_overlay(
        trees_gdf, uplvi_gdf, "UPLVI", "primary_class"
    )

    # Canopy by Naturalized vegetation type
    nat_results = compute_canopy_by_overlay(
        trees_gdf, nat_gdf, "Naturalized Areas", "vegetation_type"
    )

    # Overall city stats from classified trees
    total_trees = len(trees_gdf)
    total_canopy_ha = trees_gdf["crown_area_m2"].sum() / 1e4

    trees_only = trees_gdf[trees_gdf["veg_type"] == "tree"]
    shrubs_only = trees_gdf[trees_gdf["veg_type"] == "shrub"]

    # Trees in UPLVI vegetated areas
    uplvi_veg = uplvi_gdf[uplvi_gdf["primary_class"] == "Vegetated"]
    trees_in_natural = gpd.sjoin(trees_gdf, uplvi_veg, how="inner", predicate="within")
    trees_in_nat_areas = gpd.sjoin(trees_gdf, nat_gdf, how="inner", predicate="within")

    # Ornamental = trees NOT in natural/naturalized areas
    natural_ids = set(zip(trees_in_natural["tile"], trees_in_natural["tree_id"]))
    nat_ids = set(zip(trees_in_nat_areas["tile"], trees_in_nat_areas["tree_id"]))
    all_natural_ids = natural_ids | nat_ids

    trees_gdf["in_natural"] = list(zip(trees_gdf["tile"], trees_gdf["tree_id"]))
    ornamental = trees_gdf[~trees_gdf["in_natural"].isin(all_natural_ids)]
    natural_trees = trees_gdf[trees_gdf["in_natural"].isin(all_natural_ids)]

    log.info(f"Natural/naturalized trees: {len(natural_trees):,}")
    log.info(f"Ornamental trees: {len(ornamental):,}")

    # Build report
    report = f"""
================================================================================
  EDMONTON CANOPY COVER BY LAND TYPE — 2025 LiDAR
================================================================================

  CITY-WIDE SUMMARY
  -----------------
  Total crowns detected:     {total_trees:,}
  Total canopy area:         {total_canopy_ha:,.0f} ha

  TREES (>5m height):
    Total:                   {len(trees_only):,}
    Deciduous:               {(trees_only['leaf_type']=='deciduous').sum():,} ({100*(trees_only['leaf_type']=='deciduous').sum()/len(trees_only):.1f}%)
    Conifer:                 {(trees_only['leaf_type']=='conifer').sum():,} ({100*(trees_only['leaf_type']=='conifer').sum()/len(trees_only):.1f}%)

  SHRUBS (2-5m height):
    Total:                   {len(shrubs_only):,}
    Deciduous:               {(shrubs_only['leaf_type']=='deciduous').sum():,} ({100*(shrubs_only['leaf_type']=='deciduous').sum()/len(shrubs_only):.1f}%)
    Conifer:                 {(shrubs_only['leaf_type']=='conifer').sum():,} ({100*(shrubs_only['leaf_type']=='conifer').sum()/len(shrubs_only):.1f}%)

  BY MANAGEMENT CONTEXT
  ---------------------
  Natural / Naturalized:     {len(natural_trees):,} crowns  ({natural_trees['crown_area_m2'].sum()/1e4:,.0f} ha)
  Ornamental (planted):      {len(ornamental):,} crowns  ({ornamental['crown_area_m2'].sum()/1e4:,.0f} ha)

  UPLVI ECOCLASS BREAKDOWN
  ------------------------
"""
    if len(uplvi_results) > 0:
        for _, row in uplvi_results.iterrows():
            report += f"  {row['category']:<20s}  {row['total_crowns']:>10,} crowns  {row['total_canopy_area_ha']:>8,.0f} ha  ({row['conifer_pct']:.0f}% conifer)\n"
    else:
        report += "  No data\n"

    report += """
  NATURALIZED AREA BREAKDOWN
  --------------------------
"""
    if len(nat_results) > 0:
        for _, row in nat_results.iterrows():
            report += f"  {row['category']:<30s}  {row['total_crowns']:>8,} crowns  {row['total_canopy_area_ha']:>6,.0f} ha  ({row['conifer_pct']:.0f}% conifer)\n"
    else:
        report += "  No data\n"

    report += f"""
  FIVE-CLASS VEGETATION SUMMARY (Challenger format)
  --------------------------------------------------
  Tree Deciduous:    {(trees_only['leaf_type']=='deciduous').sum():>10,}  ({trees_only[trees_only['leaf_type']=='deciduous']['crown_area_m2'].sum()/1e4:,.0f} ha)
  Tree Coniferous:   {(trees_only['leaf_type']=='conifer').sum():>10,}  ({trees_only[trees_only['leaf_type']=='conifer']['crown_area_m2'].sum()/1e4:,.0f} ha)
  Shrub Deciduous:   {(shrubs_only['leaf_type']=='deciduous').sum():>10,}  ({shrubs_only[shrubs_only['leaf_type']=='deciduous']['crown_area_m2'].sum()/1e4:,.0f} ha)
  Shrub Coniferous:  {(shrubs_only['leaf_type']=='conifer').sum():>10,}  ({shrubs_only[shrubs_only['leaf_type']=='conifer']['crown_area_m2'].sum()/1e4:,.0f} ha)
  Grass:             (from canopy cover analysis — non-canopy low vegetation)

================================================================================
"""

    print(report)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    log.info(f"Report saved to {REPORT_FILE}")

    # Save CSVs
    if len(uplvi_results) > 0:
        uplvi_results.to_csv(os.path.join(OUTPUT_DIR, "canopy_by_uplvi.csv"), index=False)
    if len(nat_results) > 0:
        nat_results.to_csv(os.path.join(OUTPUT_DIR, "canopy_by_naturalized.csv"), index=False)

    log.info("Done!")


if __name__ == "__main__":
    main()
