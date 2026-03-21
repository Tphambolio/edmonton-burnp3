# Edmonton Urban Canopy Cover Analysis 2025

## Project Report

**City of Edmonton | Parks & Roads Services | Urban Forest Initiatives**

**Date:** March 2026
**Data Source:** 2025 City of Edmonton Vegetation LiDAR
**Program Lead:** Erin Belva, Senior Program Lead
**Technical Lead:** Travis Kennedy, P.Ag

---

## 1. Executive Summary

A comprehensive urban canopy assessment was completed using the City of Edmonton's 2025 vegetation LiDAR dataset (3,233 tiles, 480 GB, 25 pts/m2, leaf-on collection). The analysis produced city-wide canopy cover metrics, individual tree delineation, conifer/deciduous classification, top-5 species identification, and BurnP3-ready FBP fuel type mapping.

### Key Findings

| Metric | Value |
|---|---|
| **City-wide canopy cover** | **15.2%** (119.6 km2 / 784.5 km2) |
| Individual trees and shrubs detected | 5,168,142 |
| Conifer/deciduous classification accuracy | 90.4% |
| Top-5 species classification (macro F1) | 0.74 |
| Elm genus detection (F1) | 0.82 |
| Total fuel area mapped (FBP) | 35,768 ha |

Edmonton's canopy cover has increased from the 2009 UFORE baseline of 10.3% to 15.2%, representing significant progress toward the UFMP target of 20%.

---

## 2. Data Sources

| Dataset | Source | Details |
|---|---|---|
| Vegetation LiDAR | 2025 collection | 3,233 tiles, 25 pts/m2, EPSG:3776 |
| Orthophoto | Edmonton Open Data | 7.5 cm GSD RGB, 199 tiles |
| Tree Inventory | Edmonton Open Data | 474,289 city-managed trees with species |
| UPLVI | Edmonton Open Data | 12,825 ecological classification polygons |
| Naturalized Areas | Edmonton Open Data | 3,108 naturalized vegetation polygons |
| Elevation (CDEM) | NRCan | 30 m DEM, NTS sheet 083H |

---

## 3. Methodology

### 3.1 Canopy Cover Analysis

Height-normalized vegetation rasterization at 1 m resolution following USFS i-Tree / American Forests methodology:
- Ground DEM built from LAS class 2 (ground) points using binned mean statistics
- Vegetation points (classes 3, 4, 5) height-normalized by subtracting ground elevation
- Canopy defined as vegetation >= 2 m above ground (industry standard threshold)
- Rasterized to 1 m grid cells; canopy cover = canopy cells / total boundary cells
- Edmonton boundary from OpenStreetMap, reprojected to EPSG:3776

### 3.2 Individual Tree Delineation

Canopy Height Model (CHM) approach at 0.5 m resolution:
1. CHM generated as max vegetation elevation minus ground DEM
2. Gaussian smoothing (sigma = 1.5 cells) to reduce noise
3. Local maxima detection for tree top identification (minimum spacing 2 m)
4. Watershed segmentation from tree tops to delineate individual crowns
5. Per-crown feature extraction: height, area, diameter, shape metrics

### 3.3 Species Classification

#### Conifer/Deciduous (Binary)
Random Forest classifier (300 trees, max_depth=20) trained on 318,373 inventory-matched crowns with 45 LiDAR-derived features:
- Height structure: percentiles, skewness, kurtosis, vertical distribution
- RGB statistics from colorized LiDAR points
- Intensity and return structure metrics
- Crown shape: compactness, eccentricity, height-to-width ratio

Training labels derived automatically by spatial matching City tree inventory points (474K trees with genus) to LiDAR-delineated crowns within 5 m radius.

#### Top-5 Species
LightGBM classifier on 239,040 training samples with 48 features (LiDAR + orthophoto RGB):
- Classes: Ash, Aspen, Elm, Pine, Spruce
- Automated Karpathy loop (1,600+ experiments) for hyperparameter optimization

### 3.4 FBP Fuel Type Classification

CFFDRS/FBP fuel types assigned within UPLVI natural and naturalized area boundaries:
- Conifer fraction per 20 m cell computed from classified trees
- Agricultural lands differentiated using UPLVI subtype classification
- CanFG-style codes with embedded percent conifer for M-2 mixedwood

---

## 4. Results

### 4.1 Canopy Cover

| Metric | Value |
|---|---|
| City land area | 784.5 km2 (78,448 ha) |
| Total canopy area | 119.6 km2 (11,956 ha) |
| **Canopy cover** | **15.2%** |

#### Height Class Breakdown

| Class | Cover | Area |
|---|---|---|
| Low canopy (2-5 m) | 11.8% | 9,290 ha |
| Medium canopy (5-15 m) | 11.1% | 8,705 ha |
| High canopy (>15 m) | 1.9% | 1,522 ha |

#### Canopy Cover in Context

| Source | Year | Canopy % |
|---|---|---|
| UFMP / UFORE model | 2009 | 10.3% |
| **This analysis** | **2025** | **15.2%** |
| UFMP target | -- | 20% |

### 4.2 Five-Class Vegetation Summary

| Class | Count | Canopy Area (ha) |
|---|---|---|
| Tree Deciduous | 2,826,067 | 13,984 |
| Tree Coniferous | 1,491,063 | 5,553 |
| Shrub Deciduous | 681,817 | 1,254 |
| Shrub Coniferous | 169,195 | 678 |

### 4.3 By Management Context

| Context | Crowns | Canopy Area (ha) |
|---|---|---|
| Natural / Naturalized | 2,158,048 | 7,565 |
| Ornamental (planted) | 3,010,094 | 13,904 |

### 4.4 Species Classification Performance

| Species | F1 Score | Precision | Recall |
|---|---|---|---|
| Elm | 0.839 | 0.852 | 0.827 |
| Spruce | 0.788 | 0.783 | 0.793 |
| Ash | 0.787 | 0.788 | 0.786 |
| Aspen | 0.664 | 0.651 | 0.677 |
| Pine | 0.618 | 0.604 | 0.632 |

### 4.5 FBP Fuel Type Distribution

| Fuel Type | Code | Area (ha) | % of City |
|---|---|---|---|
| C-2 (Boreal Spruce) | 2 | 506 | 0.6% |
| D-2 (Green Aspen) | 12 | 2,129 | 2.7% |
| M-2 (Mixedwood Green) | 14 | 1,746 | 2.2% |
| O-1a (Crops/Pasture) | 31 | 16,984 | 21.7% |
| O-1b (Standing Grass) | 32 | 7,644 | 9.7% |
| NF (Non-fuel) | 99 | 49,434 | 63.0% |

---

## 5. Deliverables

### 5.1 GIS Layers

**GeoPackage:** `edmonton_canopy_analysis.gpkg` (93 MB)

| Layer | Type | Features |
|---|---|---|
| edmonton_boundary | Polygon | 1 |
| canopy_tiles | Polygon grid | 3,228 |
| tree_density_grid | Polygon grid | 3,228 |
| tree_crowns_500k | Points | 500,000 |

### 5.2 BurnP3/Prometheus Package

| File | Format | Description |
|---|---|---|
| fuel_type.asc + .lut + .prj | ASCII grid | FBP fuel codes (16 classes) |
| elevation.asc + .prj | ASCII grid | CDEM 30 m resampled to 20 m |
| fuel_type.tif | GeoTIFF | FBP fuel codes (for BurnP3+) |
| percent_conifer.tif | GeoTIFF | 0-100% for M-2 cells |

### 5.3 Data Files

| File | Records | Description |
|---|---|---|
| canopy_tile_results.csv | 3,228 | Per-tile canopy statistics |
| individual_trees.csv | 5,168,142 | All delineated crowns |
| trees_classified_v2.csv | 3,922,940 | Crowns with species classification |
| crown_features.csv | 3,922,940 | 42-feature profiles per crown |

### 5.4 Models

| Model | Accuracy | Description |
|---|---|---|
| classifier_v2.joblib | 90.4% | Conifer/deciduous Random Forest |
| Top-5 species (LightGBM) | 0.74 macro F1 | Elm/Ash/Spruce/Aspen/Pine |

---

## 6. Limitations and Recommendations

### 6.1 Limitations

1. **Grass classification** relies on UPLVI/naturalized boundaries rather than spectral analysis. Natural vs managed grass distinction is approximate.
2. **Species classification** accuracy varies by species. Pine (62%) and Aspen (66%) have lower F1 scores due to structural similarity with other species.
3. **Orthophoto coverage** is partial (37% of training data). Full coverage would improve species classification.
4. **Leaf-on only** -- a single collection season. Leaf-off LiDAR would improve deciduous species separation and ground classification under canopy.

### 6.2 Recommendations

1. **Acquire 4-band CIR imagery** for future collections to enable NDVI-based vegetation classification.
2. **Integrate neighbourhood features** for Elm detection to leverage spatial clustering patterns.
3. **Repeat analysis with 2019 LiDAR** for temporal change detection (canopy gain/loss).
4. **Ground-truth validation** of species classification with field crews in representative areas.
5. **Update UFMP** canopy cover target tracking with the 15.2% baseline.

---

## 7. Technical Specifications

| Parameter | Value |
|---|---|
| CRS | EPSG:3776 (NAD83 / Alberta 3TM ref merid 114 W) |
| LiDAR point density | 25 pts/m2 |
| Canopy raster resolution | 1 m |
| Tree delineation CHM | 0.5 m |
| Fuel raster cell size | 20 m |
| Canopy height threshold | 2.0 m |
| Tree/shrub boundary | 5.0 m |
| Classifier training samples | 318,373 (auto-matched from inventory) |

---

*City of Edmonton | Infrastructure Operations | Parks & Roads Services*
*Urban Forest Initiatives -- Strategy & Operations*
