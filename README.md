# Edmonton Canopy Cover Analysis

**City of Edmonton | Parks & Roads Services | Urban Forest Initiatives**

City-wide urban canopy assessment from the 2025 LiDAR dataset — individual tree delineation, conifer/deciduous classification, top-5 species identification, and BurnP3-ready FBP fuel type mapping.

## Project Information

| Field | Detail |
|---|---|
| **Department** | City Operations \| Parks & Roads Services |
| **Branch** | Urban Forest Initiatives — Strategy & Operations |
| **Program Lead** | Erin Belva, Senior Program Lead |
| **Technical Lead** | Travis Kennedy, P.Ag — General Supervisor, Urban Forestry & Ecology |
| **Data Source** | 2025 City of Edmonton Vegetation LiDAR (Contract 934295) |
| **Analysis Date** | March 2026 |
| **CRS** | EPSG:3776 (NAD83 / Alberta 3TM ref merid 114 W) |
| **Datasets (Google Drive)** | [gdrive:edmonton-canopy-analysis](https://drive.google.com/open?id=1UJ0rUjwJS5aFgYrZkSFtBCuhE6GMmX-c) |

## Key Results

| Metric | Value |
|---|---|
| **Canopy Cover** | 15.2% (119.6 km² / 784.5 km²) |
| **Trees/Shrubs Detected** | 5,168,142 |
| **Conifer/Deciduous Accuracy** | 90.4% (RF v2, 45 LiDAR features) |
| **Species Classification** | 5-class (Elm, Ash, Spruce, Aspen, Pine) — active |
| **FBP Fuel Mapping** | 35,768 ha classified for BurnP3/Prometheus |

### Canopy Cover vs UFMP Targets

| Source | Year | Canopy % |
|---|---|---|
| UFMP / UFORE model | 2009 | 10.3% |
| **This analysis (LiDAR)** | **2025** | **15.2%** |
| UFMP target | — | 20% |

### Five-Class Vegetation Summary

| Class | Count | Canopy Area (ha) |
|---|---|---|
| Tree Deciduous | 2,826,067 | 13,984 |
| Tree Coniferous | 1,491,063 | 5,553 |
| Shrub Deciduous | 681,817 | 1,254 |
| Shrub Coniferous | 169,195 | 678 |

### FBP Fuel Type Distribution (Natural Areas)

| Fuel Type | Area (ha) | Description |
|---|---|---|
| C-2 | 506 | Boreal Spruce |
| D-2 | 2,129 | Green Aspen |
| M-2 | 1,746 | Boreal Mixedwood (with PC%) |
| O-1a | 16,984 | Crops / Pasture |
| O-1b | 7,644 | Standing Natural Grass |
| NF | 49,434 | Non-fuel (urban) |

## Data Sources

| Dataset | Source | Records |
|---|---|---|
| Vegetation LiDAR | 2025 collection, 25 pts/m², leaf-on | 3,233 tiles (480 GB) |
| Orthophoto | Edmonton Open Data (7.5 cm GSD RGB) | 199 tiles (56 GB) |
| Tree Inventory | Edmonton Open Data (`eecg-fc54`) | 474,289 trees |
| UPLVI | Edmonton Open Data (`iu6b-n5sr`) | 12,825 polygons |
| Naturalized Areas | Edmonton Open Data (`ct88-r8tk`) | 3,108 polygons |
| Elevation (CDEM) | NRCan 083H | 30m DEM |

## Pipeline

```
LiDAR (3,233 LAZ tiles, 480 GB)
  │
  ├── scripts/canopy_cover_analysis.py    → 15.2% canopy cover (3,228 tiles)
  ├── scripts/tree_delineation.py         → 5.17M individual crowns
  ├── scripts/extract_features_v2.py      → 42 per-crown features (3.92M crowns)
  ├── scripts/species_classifier.py       → conifer/deciduous v1 (83.3%)
  ├── scripts/retrain_classifier.py       → conifer/deciduous v2 (90.4%)
  ├── scripts/canopy_by_landtype.py       → UPLVI + naturalized overlay
  ├── scripts/build_fuel_raster.py        → FBP fuel type raster (20m)
  ├── scripts/export_burnp3.py            → Prometheus ASCII grids + LUT
  ├── scripts/export_gis_layers.py        → GeoPackage (93 MB, 4 layers)
  │
  └── tree-autoresearch/                  → Karpathy loop (species)
      ├── prepare.py                      → spatial match inventory → LiDAR + ortho
      ├── autoresearch.py                 → automated mutation/eval loop
      └── train.py                        → best model (LightGBM)
```

## Species Classification (Karpathy Loop)

Automated experiment loop at `/home/rpas/dev/tree-autoresearch/`:
- **Training data:** 239,040 inventory-matched trees (48 features: LiDAR structure + RGB + ortho)
- **Classes:** Ash (92K), Elm (92K), Spruce (66K), Pine (30K), Aspen (19K)
- **Method:** Random mutation of model config + feature engineering → eval → keep/revert
- **Experiments:** 1,600+ completed, ongoing

## GIS Outputs

### GeoPackage: `edmonton_canopy_analysis.gpkg` (93 MB)

| Layer | Type | Features | Key Attributes |
|---|---|---|---|
| edmonton_boundary | Polygon | 1 | canopy_cover_pct, total_area_km2 |
| canopy_tiles | Polygon grid | 3,228 | canopy_pct, low/med/high_pct |
| tree_density_grid | Polygon grid | 3,228 | tree_count, trees_per_ha, mean_height |
| trees_sample_500k | Points | 500,000 | max_height, crown_area, leaf_type, conifer_prob |

### BurnP3/Prometheus Package: `fuel_raster/prometheus/`

| File | Description |
|---|---|
| `fuel_type.asc` + `.lut` + `.prj` | FBP fuel codes (16 classes, CanFG M-2 with embedded PC%) |
| `elevation.asc` + `.prj` | CDEM 30m resampled to 20m |

Load directly into Prometheus: import fuel_type.asc → import elevation.asc → go.

## Setup

```bash
# Dependencies
pip install laspy scipy scikit-learn scikit-image rasterio geopandas \
    shapely pyproj lightgbm xgboost joblib

# Download datasets from Google Drive
rclone copy gdrive:edmonton-canopy-analysis/data/ data/
rclone copy gdrive:edmonton-canopy-analysis/outputs/ outputs/

# Run full pipeline
python scripts/canopy_cover_analysis.py              # canopy cover
python scripts/tree_delineation.py --workers 8        # individual trees
python scripts/extract_features_v2.py                 # per-crown features
python scripts/retrain_classifier.py                  # conifer/deciduous v2
python scripts/canopy_by_landtype.py                  # land type overlay
python scripts/build_fuel_raster.py                   # FBP fuel raster
python scripts/export_burnp3.py                       # Prometheus ASCII
python scripts/export_gis_layers.py                   # GeoPackage

# Species classification loop
cd tree-autoresearch && python autoresearch.py 500
```

## Related Projects

| Project | GitLab Path | Description |
|---|---|---|
| FlightHub 2 MCP | `opm.../edmonton-rpas-flighthub-2` | DJI FlightHub 2 plugin |
| LiDAR Tree Loader | `opm.../edmonton-lidar-tree-loader` | LiDAR tree ingestion pipeline |
| Tree CAD Extract | `opm.../tree-cad-extract` | CAD tree extraction tool |

## License

Internal use — City of Edmonton, Infrastructure Operations, Parks & Roads Services.
