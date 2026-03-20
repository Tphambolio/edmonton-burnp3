#!/usr/bin/env python3
"""
Export BurnP3/Prometheus-Ready Fuel and Elevation Grids
========================================================
Converts our fuel type GeoTIFF into the exact format Prometheus expects:
  - fuel_type.asc + .lut + .prj  (ASCII grid with lookup table)
  - elevation.asc + .prj         (DEM from LiDAR ground points)

M-2 mixedwood cells get CanFG-style codes with embedded percent conifer.
"""

import os
import logging
import time
import glob
import numpy as np
import rasterio
from rasterio.transform import from_bounds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("burnp3")

BASE_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"
FUEL_TIF = os.path.join(BASE_DIR, "fuel_raster/fuel_type.tif")
PC_TIF = os.path.join(BASE_DIR, "fuel_raster/percent_conifer.tif")
OUTPUT_DIR = os.path.join(BASE_DIR, "fuel_raster/prometheus")

NODATA = -9999

# EPSG:3776 WKT for .prj file
PRJ_WKT = (
    'PROJCS["NAD83 / Alberta 3TM ref merid 114 W",'
    'GEOGCS["NAD83",'
    'DATUM["North_American_Datum_1983",'
    'SPHEROID["GRS 1980",6378137,298.257222101]],'
    'PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",-114],'
    'PARAMETER["scale_factor",0.9999],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1]]'
)

# CanFG-style codes for M-2 with embedded percent conifer
# Format: 5XX where XX is percent conifer (rounded to nearest 5)
CANFG_M2_BASE = 500  # M-2 Green starts at 500


def reclassify_mixedwood(fuel, pc):
    """
    Replace M-2 (code 14) cells with CanFG-style codes embedding PC%.
    525 = M-2 at 25% conifer, 550 = M-2 at 50%, etc.
    """
    m2_mask = fuel == 14
    m2_count = m2_mask.sum()
    log.info(f"  Reclassifying {m2_count:,} M-2 cells with embedded PC%...")

    # Bin percent conifer to nearest 5%
    pc_binned = np.clip(np.round(pc[m2_mask] / 5) * 5, 5, 95).astype(int)
    fuel[m2_mask] = CANFG_M2_BASE + pc_binned

    # Report distribution
    unique, counts = np.unique(fuel[m2_mask], return_counts=True)
    for code, count in zip(unique, counts):
        pc_val = code - CANFG_M2_BASE
        log.info(f"    M-2 C{pc_val:02d} (code {code}): {count:,} cells")

    return fuel


def write_ascii_grid(data, path, xllcorner, yllcorner, cellsize, nodata=NODATA):
    """Write ESRI ASCII grid file."""
    ny, nx = data.shape
    with open(path, "w") as f:
        f.write(f"ncols         {nx}\n")
        f.write(f"nrows         {ny}\n")
        f.write(f"xllcorner     {xllcorner}\n")
        f.write(f"yllcorner     {yllcorner}\n")
        f.write(f"cellsize      {cellsize}\n")
        f.write(f"NODATA_value  {nodata}\n")
        for row in data:
            f.write(" ".join(str(int(v)) for v in row) + "\n")
    log.info(f"  Written: {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def write_ascii_grid_float(data, path, xllcorner, yllcorner, cellsize, nodata=NODATA):
    """Write ESRI ASCII grid with float values (for DEM)."""
    ny, nx = data.shape
    with open(path, "w") as f:
        f.write(f"ncols         {nx}\n")
        f.write(f"nrows         {ny}\n")
        f.write(f"xllcorner     {xllcorner}\n")
        f.write(f"yllcorner     {yllcorner}\n")
        f.write(f"cellsize      {cellsize}\n")
        f.write(f"NODATA_value  {nodata}\n")
        for row in data:
            f.write(" ".join(f"{v:.1f}" if v != nodata else str(nodata) for v in row) + "\n")
    log.info(f"  Written: {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def write_lut(fuel_data, path):
    """Write Prometheus fuel lookup table (.lut)."""
    unique_codes = sorted(np.unique(fuel_data[fuel_data != NODATA]))

    lut_map = {
        2: "C-2",
        3: "C-3",
        12: "D-2",
        31: "O-1a",
        32: "O-1b",
        98: "Water",
        99: "Non Fuel",
    }
    # Add CanFG M-2 codes
    for code in unique_codes:
        if CANFG_M2_BASE < code < CANFG_M2_BASE + 100:
            pc = code - CANFG_M2_BASE
            lut_map[code] = f"M-2"  # Prometheus M-2 with PC override

    with open(path, "w") as f:
        for code in unique_codes:
            if code == NODATA or code == 0:
                continue
            name = lut_map.get(code, f"Unknown_{code}")
            # Prometheus LUT format: code,FBP_name[,percent_conifer]
            if CANFG_M2_BASE < code < CANFG_M2_BASE + 100:
                pc = code - CANFG_M2_BASE
                f.write(f"{code},{name},{pc}\n")
            else:
                f.write(f"{code},{name}\n")

    log.info(f"  Written: {path}")
    log.info(f"  Fuel types in LUT: {len(unique_codes)}")


def write_prj(path):
    """Write projection file."""
    with open(path, "w") as f:
        f.write(PRJ_WKT)
    log.info(f"  Written: {path}")


def build_dem_from_canopy_tiles():
    """
    Build a 20m DEM from the canopy analysis tile results.
    Uses the ground elevation data already computed per tile.
    """
    log.info("Building 20m DEM from LiDAR ground points...")

    # Read the fuel raster to get the target extent
    with rasterio.open(FUEL_TIF) as src:
        bounds = src.bounds
        nx, ny = src.width, src.height
        cellsize = src.res[0]

    xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top

    # We need to read canopy tile results for the ground elevation
    # The tile CSV has xmin/ymin/xmax/ymax per tile
    import pandas as pd
    tiles = pd.read_csv(os.path.join(BASE_DIR, "canopy_tile_results.csv"))

    # Build DEM by reading ground points from LAZ tiles
    # For efficiency, use binned statistics at 20m from the LAZ ground points
    import laspy
    from scipy.stats import binned_statistic_2d
    from scipy.ndimage import distance_transform_edt

    dem = np.full((ny, nx), np.nan, dtype=np.float32)

    LAZ_DIR = os.path.join(BASE_DIR, "edmonton-veglidar-laz")
    laz_files = sorted(glob.glob(os.path.join(LAZ_DIR, "*.laz")))

    x_edges = np.arange(xmin, xmax + cellsize, cellsize)
    y_edges = np.arange(ymin, ymax + cellsize, cellsize)

    processed = 0
    for laz_path in laz_files:
        try:
            las = laspy.read(laz_path)
        except Exception:
            continue

        ground_mask = las.classification == 2
        gx = las.x[ground_mask]
        gy = las.y[ground_mask]
        gz = las.z[ground_mask]

        if len(gx) < 10:
            continue

        # Compute column/row indices for these ground points in the DEM grid
        cols = ((gx - xmin) / cellsize).astype(int)
        rows = ny - 1 - ((gy - ymin) / cellsize).astype(int)

        valid = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
        cols, rows, gz_v = cols[valid], rows[valid], gz[valid]

        # Mean ground elevation per cell
        for c, r, z in zip(cols, rows, gz_v):
            if np.isnan(dem[r, c]):
                dem[r, c] = z
            else:
                dem[r, c] = (dem[r, c] + z) / 2  # running average approximation

        processed += 1
        if processed % 200 == 0:
            filled = np.sum(~np.isnan(dem))
            log.info(f"    {processed} tiles, {filled:,}/{ny*nx:,} cells filled ({100*filled/(ny*nx):.0f}%)")

        del las
        import gc
        gc.collect()

    # Fill remaining NaN cells with nearest neighbor
    nan_mask = np.isnan(dem)
    filled_before = np.sum(~nan_mask)
    log.info(f"  DEM cells filled: {filled_before:,}/{ny*nx:,} ({100*filled_before/(ny*nx):.0f}%)")

    if nan_mask.any() and not nan_mask.all():
        indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        dem = dem[tuple(indices)]
        log.info(f"  NaN cells filled with nearest neighbor")

    return dem, xmin, ymin, cellsize


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Export BurnP3/Prometheus-Ready Grids")
    log.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Read fuel type and percent conifer rasters
    log.info("Reading fuel type raster...")
    with rasterio.open(FUEL_TIF) as src:
        fuel = src.read(1)
        bounds = src.bounds
        cellsize = src.res[0]
        nx, ny = src.width, src.height

    xmin, ymin = bounds.left, bounds.bottom

    log.info("Reading percent conifer raster...")
    with rasterio.open(PC_TIF) as src:
        pc = src.read(1)

    # Step 2: Reclassify M-2 with embedded PC%
    fuel = reclassify_mixedwood(fuel, pc)

    # Step 3: Set nodata
    fuel[fuel == 0] = NODATA

    # Step 4: Write fuel ASCII grid + LUT + PRJ
    fuel_asc = os.path.join(OUTPUT_DIR, "fuel_type.asc")
    fuel_lut = os.path.join(OUTPUT_DIR, "fuel_type.lut")
    fuel_prj = os.path.join(OUTPUT_DIR, "fuel_type.prj")

    log.info("Writing fuel type ASCII grid...")
    write_ascii_grid(fuel, fuel_asc, xmin, ymin, cellsize)
    write_lut(fuel, fuel_lut)
    write_prj(fuel_prj)

    # Step 5: Build and write elevation DEM
    dem, dem_xmin, dem_ymin, dem_cellsize = build_dem_from_canopy_tiles()

    elev_asc = os.path.join(OUTPUT_DIR, "elevation.asc")
    elev_prj = os.path.join(OUTPUT_DIR, "elevation.prj")

    # Set nodata for DEM where outside boundary (match fuel nodata locations)
    dem[fuel == NODATA] = NODATA

    log.info("Writing elevation ASCII grid...")
    write_ascii_grid_float(dem, elev_asc, dem_xmin, dem_ymin, dem_cellsize)
    write_prj(elev_prj)

    # Step 6: Also write percent_conifer.asc for BurnP3+
    pc_asc = os.path.join(OUTPUT_DIR, "percent_conifer.asc")
    pc_out = pc.copy().astype(np.int16)
    pc_out[fuel == NODATA] = NODATA
    write_ascii_grid(pc_out, pc_asc, xmin, ymin, cellsize)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"Complete in {elapsed:.0f}s")
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1e6
        log.info(f"  {f} ({size:.1f} MB)")
    log.info("")
    log.info("To load in Prometheus:")
    log.info("  1. Import fuel_type.asc (auto-loads .lut and .prj)")
    log.info("  2. Import elevation.asc (auto-loads .prj)")
    log.info("  3. Prometheus derives slope/aspect from elevation")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
