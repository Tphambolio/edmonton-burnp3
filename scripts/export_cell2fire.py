#!/usr/bin/env python3
"""
Export Cell2Fire-compatible inputs from Edmonton fuel raster.
Generates: Forest.asc, Data.csv, fbp_lookup_table.csv, Weather.csv template
"""
import os, logging, numpy as np, rasterio
from rasterio.transform import from_bounds, rowcol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("c2f")

BASE = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"
FUEL_TIF = os.path.join(BASE, "fuel_raster/fuel_type.tif")
ELEV_TIF = os.path.join(BASE, "fuel_raster/prometheus/elevation.tif")
OUT = os.path.join(BASE, "fuel_raster/cell2fire")

# FBP code mapping (grid integer → Cell2Fire string)
FBP_MAP = {
    2: ("C2", "Boreal Spruce", 34, 102, 51),
    12: ("D2", "Green Aspen", 255, 211, 127),
    31: ("O1a", "Matted Grass", 255, 255, 115),
    32: ("O1b", "Standing Grass", 230, 230, 0),
    99: ("NF", "Non-fuel", 178, 178, 178),
    98: ("WA", "Water", 0, 112, 255),
}
# CanFG M-2 codes (500-599)
for pc in range(5, 100, 5):
    FBP_MAP[500 + pc] = (f"M2", f"Mixedwood Green {pc}%", 168, 168, 0)


def main():
    os.makedirs(OUT, exist_ok=True)
    log.info("=== Cell2Fire Export ===")

    # Read fuel raster
    with rasterio.open(FUEL_TIF) as src:
        fuel = src.read(1)
        transform = src.transform
        ny, nx = fuel.shape
        cellsize = src.res[0]
        bounds = src.bounds

    # Read elevation
    with rasterio.open(ELEV_TIF) as src:
        elev = src.read(1)

    log.info(f"Grid: {nx}x{ny}, cellsize={cellsize}m")

    # 1. Forest.asc — already have this, but Cell2Fire needs sequential integer IDs
    # Remap our codes to sequential integers for the lookup table
    unique_codes = sorted(set(fuel[fuel != -9999].ravel()))
    code_to_id = {c: i + 1 for i, c in enumerate(unique_codes)}
    code_to_id[-9999] = -9999

    forest = np.vectorize(lambda x: code_to_id.get(x, -9999))(fuel)

    forest_path = os.path.join(OUT, "Forest.asc")
    with open(forest_path, "w") as f:
        f.write(f"ncols         {nx}\n")
        f.write(f"nrows         {ny}\n")
        f.write(f"xllcorner     {bounds.left}\n")
        f.write(f"yllcorner     {bounds.bottom}\n")
        f.write(f"cellsize      {cellsize}\n")
        f.write(f"NODATA_value  -9999\n")
        for row in forest:
            f.write(" ".join(str(int(v)) for v in row) + "\n")
    log.info(f"Forest.asc: {os.path.getsize(forest_path)/1e6:.1f} MB")

    # 2. fbp_lookup_table.csv
    lut_path = os.path.join(OUT, "fbp_lookup_table.csv")
    with open(lut_path, "w") as f:
        f.write("grid_value,export_value,descriptive_name,fuel_type,r,g,b,h,s,l\n")
        for orig_code, seq_id in sorted(code_to_id.items(), key=lambda x: x[1]):
            if orig_code == -9999:
                continue
            if orig_code in FBP_MAP:
                ftype, desc, r, g, b = FBP_MAP[orig_code]
            else:
                ftype, desc, r, g, b = "NF", f"Unknown_{orig_code}", 130, 130, 130
            f.write(f"{seq_id},{seq_id},{desc},{ftype},{r},{g},{b},0,0,0\n")
    log.info(f"fbp_lookup_table.csv: {sum(1 for _ in open(lut_path)) - 1} entries")

    # 3. Data.csv — one row per cell, row-major order
    log.info("Building Data.csv (one row per cell)...")
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:3776", "EPSG:4326", always_xy=True)

    data_path = os.path.join(OUT, "Data.csv")
    with open(data_path, "w") as f:
        f.write("fueltype,mon,jd,M,jd_min,lat,lon,elev,ffmc,ws,waz,bui,ps,saz,pc,pdf,gfl,cur,time,pattern\n")
        for row_i in range(ny):
            for col_i in range(nx):
                fcode = fuel[row_i, col_i]
                if fcode == -9999 or fcode == 0:
                    ftype = "NF"
                    pc = 0
                elif fcode in FBP_MAP:
                    ftype = FBP_MAP[fcode][0]
                    pc = fcode - 500 if 500 < fcode < 600 else 0
                else:
                    ftype = "NF"
                    pc = 0

                # Cell center coordinates
                cx = bounds.left + (col_i + 0.5) * cellsize
                cy = bounds.top - (row_i + 0.5) * cellsize
                lon, lat = t.transform(cx, cy)
                e = float(elev[row_i, col_i]) if elev[row_i, col_i] != -9999 else 0

                # Grass fuel load and curing
                gfl = 0.35 if ftype in ("O1a", "O1b") else 0
                cur = 60 if ftype == "O1a" else (30 if ftype == "O1b" else 0)

                f.write(f"{ftype},,,,,{lat:.6f},{lon:.6f},{e:.1f},,,,,,{0},{pc},{0},{gfl},{cur},20,\n")

            if (row_i + 1) % 200 == 0:
                log.info(f"  Row {row_i+1}/{ny}")

    log.info(f"Data.csv: {os.path.getsize(data_path)/1e6:.1f} MB, {ny*nx:,} cells")

    # 4. Weather.csv template
    weather_path = os.path.join(OUT, "Weather.csv")
    with open(weather_path, "w") as f:
        f.write("scenario,datetime,apcp,tmp,rh,ws,waz,ffmc,dmc,dc,isi,bui,fwi\n")
        # Sample summer fire weather for Edmonton (July high-risk day)
        for hour in range(24):
            tmp = 28 if 10 <= hour <= 18 else 15
            rh = 25 if 10 <= hour <= 18 else 55
            ws = 20 if 12 <= hour <= 17 else 10
            waz = 270  # westerly
            f.write(f"1,2025-07-15 {hour:02d}:00,0,{tmp},{rh},{ws},{waz},89,50,300,12,80,25\n")
    log.info(f"Weather.csv: template with 24 hourly records")

    # 5. Elevation ASC (copy)
    import shutil
    elev_src = os.path.join(BASE, "fuel_raster/prometheus/elevation.asc")
    if os.path.exists(elev_src):
        shutil.copy(elev_src, os.path.join(OUT, "elevation.asc"))
        log.info("elevation.asc: copied from Prometheus export")

    log.info(f"\n=== Cell2Fire package ready: {OUT} ===")
    for f in sorted(os.listdir(OUT)):
        log.info(f"  {f}: {os.path.getsize(os.path.join(OUT, f))/1e6:.1f} MB")


if __name__ == "__main__":
    main()
