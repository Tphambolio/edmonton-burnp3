# Edmonton Canopy Cover Analysis

City-wide urban canopy assessment from the 2025 LiDAR dataset — individual tree delineation, conifer/deciduous classification, and top-5 species identification.

## Key Results

| Metric | Value |
|---|---|
| **Canopy Cover** | 15.2% (119.6 km² / 784.5 km²) |
| **Trees/Shrubs Detected** | 5,168,142 |
| **Conifer/Deciduous Accuracy** | 90.4% (v2, 45 LiDAR features) |
| **Species Classification** | 5-class (Elm, Ash, Spruce, Aspen, Pine) — Karpathy loop active |

## Data

- **LiDAR:** 2025 City of Edmonton vegetation LiDAR (25 pts/m², leaf-on, EPSG:3776)
- **Orthophoto:** 2025 RGB orthophoto (7.5 cm GSD, Edmonton Open Data)
- **Tree Inventory:** 474,289 city-managed trees with species labels (Edmonton Open Data)
- **UPLVI:** Urban Primary Land and Vegetation Inventory ecoclasses
- **Naturalized Areas:** 3,108 naturalized vegetation polygons

Large datasets stored on Google Drive: `gdrive:edmonton-canopy-analysis/`

## Pipeline

```
LiDAR (3,233 LAZ tiles, 480 GB)
  ├── canopy_cover_analysis.py     → 15.2% canopy cover
  ├── tree_delineation.py          → 5.17M individual crowns
  ├── extract_features_v2.py       → 42 per-crown LiDAR features (3.92M)
  ├── species_classifier.py        → conifer/deciduous (90.4%)
  ├── canopy_by_landtype.py        → UPLVI + naturalized overlay
  └── tree-autoresearch/
      ├── prepare.py               → spatial match inventory to LiDAR+ortho
      ├── autoresearch.py          → Karpathy loop (species classification)
      └── train.py                 → best model (mutated by loop)
```

## Species Classification (Karpathy Loop)

Automated experiment loop mutating model configs and feature engineering:
- **Training data:** 239,040 inventory-matched trees (48 features: LiDAR + ortho RGB)
- **Classes:** Ash, Aspen, Elm, Pine, Spruce
- **Loop:** `autoresearch.py` — random mutation + eval, keep improvements, revert regressions
- **Current best:** val_f1 0.7369 (on expanded 239K dataset)

## Setup

```bash
# Dependencies
pip install laspy scipy scikit-learn scikit-image rasterio geopandas shapely pyproj lightgbm xgboost

# Download datasets from Google Drive
rclone copy gdrive:edmonton-canopy-analysis/data/ data/

# Run canopy cover analysis
python scripts/canopy_cover_analysis.py

# Run tree delineation
python scripts/tree_delineation.py --workers 8

# Run species classification loop
cd tree-autoresearch && python autoresearch.py 500
```

## Outputs

| File | Description |
|---|---|
| `outputs/canopy_cover_report.txt` | City-wide canopy cover summary |
| `outputs/canopy_tile_results.csv` | Per-tile canopy statistics (3,228 tiles) |
| `outputs/individual_trees.csv` | 5.17M delineated crowns |
| `outputs/trees_classified_v2.csv` | Crowns with conifer/deciduous + features |
| `outputs/canopy_by_landtype_report.txt` | Overlay analysis by land type |
| `outputs/classification_report_v2.txt` | Classifier performance metrics |

## License

Internal — City of Edmonton, Parks & Roads Services
