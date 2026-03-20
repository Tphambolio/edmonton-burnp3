#!/usr/bin/env python3
"""
Edmonton Conifer/Deciduous Tree Classification
===============================================
Uses the City's 474K tree inventory (with species/genus) as training labels,
spatially matched to LiDAR-delineated crowns, to train a random forest
classifier on per-crown LiDAR features.

Pipeline:
1. Load tree inventory → classify genus as conifer/deciduous
2. Reproject inventory points to EPSG:3776
3. Spatially join inventory points to delineated crowns (nearest within 5m)
4. Extract LiDAR-based features for matched crowns
5. Train random forest classifier
6. Predict for all 5.17M crowns
"""

import os
import csv
import json
import time
import logging
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("species")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"
INVENTORY_FILE = os.path.join(BASE_DIR, "edmonton_tree_inventory.json")
TREES_CSV = os.path.join(BASE_DIR, "tree_results/individual_trees.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "tree_results/trees_classified.csv")
MODEL_REPORT = os.path.join(BASE_DIR, "tree_results/classification_report.txt")

MATCH_RADIUS = 5.0  # metres — max distance to match inventory to crown

# Genus → conifer/deciduous lookup
CONIFER_GENERA = {
    "Picea",        # Spruce (69,704)
    "Pinus",        # Pine (32,183)
    "Larix",        # Larch (9,817) — deciduous conifer but classified as conifer structurally
    "Pseudotsuga",  # Douglas Fir (688)
    "Thuja",        # Cedar
    "Juniperus",    # Juniper
    "Abies",        # Fir
    "Tsuga",        # Hemlock
    "Taxus",        # Yew
}

DECIDUOUS_GENERA = {
    "Ulmus",        # Elm (96,682)
    "Fraxinus",     # Ash (95,304)
    "Populus",      # Poplar/Aspen (40,005)
    "Malus",        # Apple/Crabapple (24,955)
    "Quercus",      # Oak (19,476)
    "Acer",         # Maple (17,502)
    "Tilia",        # Linden (15,688)
    "Prunus",       # Cherry/Plum (14,828)
    "Syringa",      # Lilac (9,560)
    "Sorbus",       # Mountain Ash (6,954)
    "Salix",        # Willow (5,987)
    "Betula",       # Birch (4,094)
    "Aesculus",     # Horse Chestnut (3,394)
    "Crataegus",    # Hawthorn (2,965)
    "Eleagnus",     # Oleaster (1,392)
    "Pyrus",        # Pear (839)
    "Cornus",       # Dogwood
    "Amelanchier",  # Saskatoon
    "Caragana",     # Caragana
    "Juglans",      # Walnut
    "Sambucus",     # Elder
}


# ---------------------------------------------------------------------------
# Step 1: Load & classify inventory
# ---------------------------------------------------------------------------
def load_inventory():
    """Load tree inventory and classify as conifer/deciduous."""
    log.info("Loading tree inventory...")
    with open(INVENTORY_FILE) as f:
        raw = json.load(f)

    trees = []
    for t in raw:
        genus = t.get("genus", "")
        lat = t.get("latitude")
        lon = t.get("longitude")
        if not lat or not lon or not genus:
            continue

        if genus in CONIFER_GENERA:
            leaf_type = "conifer"
        elif genus in DECIDUOUS_GENERA:
            leaf_type = "deciduous"
        else:
            continue  # skip unknown genera

        trees.append({
            "genus": genus,
            "species": t.get("species", ""),
            "leaf_type": leaf_type,
            "lat": float(lat),
            "lon": float(lon),
            "dbh": float(t["diameter_breast_height"]) if t.get("diameter_breast_height") else None,
        })

    log.info(f"Loaded {len(trees):,} classifiable trees")
    conifer_count = sum(1 for t in trees if t["leaf_type"] == "conifer")
    decid_count = sum(1 for t in trees if t["leaf_type"] == "deciduous")
    log.info(f"  Conifer: {conifer_count:,}  Deciduous: {decid_count:,}")

    return trees


# ---------------------------------------------------------------------------
# Step 2: Reproject inventory to EPSG:3776
# ---------------------------------------------------------------------------
def reproject_inventory(trees):
    """Convert inventory lat/lon to EPSG:3776."""
    log.info("Reprojecting inventory to EPSG:3776...")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3776", always_xy=True)

    for t in trees:
        x, y = transformer.transform(t["lon"], t["lat"])
        t["x"] = x
        t["y"] = y

    return trees


# ---------------------------------------------------------------------------
# Step 3: Load delineated crowns and spatial match
# ---------------------------------------------------------------------------
def load_crowns():
    """Load the delineated crown CSV."""
    log.info("Loading delineated crowns...")
    df = pd.read_csv(TREES_CSV)
    log.info(f"Loaded {len(df):,} crowns")
    return df


def spatial_match(inventory, crowns_df, radius=MATCH_RADIUS):
    """
    Match inventory points to nearest delineated crown centroid.
    Returns DataFrame of matched crowns with leaf_type label.
    """
    log.info(f"Spatial matching (radius={radius}m)...")

    # Build KD-tree from crown centroids
    crown_xy = crowns_df[["centroid_x", "centroid_y"]].values
    tree = cKDTree(crown_xy)

    # Query inventory points
    inv_xy = np.array([[t["x"], t["y"]] for t in inventory])
    distances, indices = tree.query(inv_xy, k=1)

    # Filter by radius
    matched = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist <= radius:
            crown_row = crowns_df.iloc[idx].to_dict()
            crown_row["leaf_type"] = inventory[i]["leaf_type"]
            crown_row["inv_genus"] = inventory[i]["genus"]
            crown_row["inv_species"] = inventory[i]["species"]
            crown_row["match_dist"] = round(dist, 2)
            matched.append(crown_row)

    log.info(f"Matched {len(matched):,} inventory trees to crowns "
             f"({len(matched)/len(inventory)*100:.1f}% match rate)")

    matched_df = pd.DataFrame(matched)

    # Deduplicate: if multiple inventory points match the same crown, keep closest
    if "tree_id" in matched_df.columns and "tile" in matched_df.columns:
        matched_df = matched_df.sort_values("match_dist")
        matched_df = matched_df.drop_duplicates(subset=["tile", "tree_id"], keep="first")
        log.info(f"After dedup: {len(matched_df):,} unique crown matches")

    conifer_count = (matched_df["leaf_type"] == "conifer").sum()
    decid_count = (matched_df["leaf_type"] == "deciduous").sum()
    log.info(f"  Conifer: {conifer_count:,}  Deciduous: {decid_count:,}")

    return matched_df


# ---------------------------------------------------------------------------
# Step 4: Feature engineering for classification
# ---------------------------------------------------------------------------
def prepare_features(df):
    """
    Prepare feature matrix from crown attributes.
    LiDAR-based features that distinguish conifers from deciduous:
    - Height statistics (conifers tend taller, narrower)
    - Crown shape (conifers more compact/circular, deciduous more spread)
    - Height variance (conifers more uniform vertical structure)
    """
    features = pd.DataFrame()

    # Height features
    features["max_height"] = df["max_height"]
    features["mean_height"] = df["mean_height"]
    features["std_height"] = df["std_height"]
    features["height_cv"] = df["std_height"] / df["mean_height"].clip(lower=0.1)

    # Crown shape features
    features["crown_area"] = df["crown_area_m2"]
    features["crown_diameter"] = df["crown_diameter_m"]
    features["compactness"] = df["compactness"]
    features["eccentricity"] = df["eccentricity"]

    # Derived ratios
    features["height_to_crown_ratio"] = df["max_height"] / df["crown_diameter_m"].clip(lower=0.1)
    features["height_to_area_ratio"] = df["max_height"] / df["crown_area_m2"].clip(lower=0.1)
    features["perimeter_to_area"] = df["perimeter_m"] / df["crown_area_m2"].clip(lower=0.1)

    # Replace infinities and NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    return features


# ---------------------------------------------------------------------------
# Step 5: Train & evaluate classifier
# ---------------------------------------------------------------------------
def train_classifier(matched_df):
    """Train random forest on matched crowns."""
    log.info("Preparing features...")
    X = prepare_features(matched_df)
    y = (matched_df["leaf_type"] == "conifer").astype(int)  # 1=conifer, 0=deciduous

    log.info(f"Feature matrix: {X.shape}")
    log.info(f"Class balance: conifer={y.sum():,}, deciduous={(~y.astype(bool)).sum():,}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Train
    log.info("Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=["deciduous", "conifer"],
    )
    cm = confusion_matrix(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    # Feature importance
    importances = sorted(
        zip(X.columns, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )

    log.info(f"\n{report}")
    log.info(f"Cross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    log.info(f"Confusion matrix:\n{cm}")

    # Save report
    report_text = f"""
================================================================================
  CONIFER / DECIDUOUS CLASSIFICATION REPORT
================================================================================

  Training Data:  {len(matched_df):,} inventory-matched crowns
  Train/Test:     {len(X_train):,} / {len(X_test):,}
  Model:          Random Forest (200 trees, max_depth=15)
  Features:       LiDAR-derived crown metrics (no orthophoto)

  PERFORMANCE
  -----------
{report}

  Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}

  Confusion Matrix:
                 Pred Deciduous  Pred Conifer
  Act Deciduous    {cm[0][0]:>10,}   {cm[0][1]:>10,}
  Act Conifer      {cm[1][0]:>10,}   {cm[1][1]:>10,}

  Feature Importance:
"""
    for fname, imp in importances:
        report_text += f"    {fname:<25s} {imp:.4f}\n"

    report_text += "\n================================================================================\n"

    with open(MODEL_REPORT, "w") as f:
        f.write(report_text)
    log.info(f"Report saved to {MODEL_REPORT}")

    return clf, X.columns.tolist()


# ---------------------------------------------------------------------------
# Step 6: Predict for all crowns
# ---------------------------------------------------------------------------
def predict_all(clf, feature_names, crowns_df):
    """Classify all delineated crowns as conifer/deciduous."""
    log.info(f"Predicting for all {len(crowns_df):,} crowns...")

    X_all = prepare_features(crowns_df)
    # Ensure same column order
    X_all = X_all[feature_names]

    predictions = clf.predict(X_all)
    probabilities = clf.predict_proba(X_all)

    crowns_df = crowns_df.copy()
    crowns_df["leaf_type"] = np.where(predictions == 1, "conifer", "deciduous")
    crowns_df["conifer_prob"] = probabilities[:, 1]
    crowns_df["deciduous_prob"] = probabilities[:, 0]

    conifer_count = (crowns_df["leaf_type"] == "conifer").sum()
    decid_count = (crowns_df["leaf_type"] == "deciduous").sum()
    log.info(f"Predictions: conifer={conifer_count:,}, deciduous={decid_count:,}")
    log.info(f"Conifer %: {100*conifer_count/len(crowns_df):.1f}%")

    return crowns_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("Edmonton Conifer/Deciduous Classification")
    log.info("=" * 60)

    # Load data
    inventory = load_inventory()
    inventory = reproject_inventory(inventory)
    crowns_df = load_crowns()

    # Spatial match
    matched_df = spatial_match(inventory, crowns_df)

    if len(matched_df) < 100:
        log.error("Too few matches to train classifier")
        return

    # Train
    clf, feature_names = train_classifier(matched_df)

    # Predict all
    classified_df = predict_all(clf, feature_names, crowns_df)

    # Save
    log.info(f"Saving classified trees to {OUTPUT_CSV}")
    classified_df.to_csv(OUTPUT_CSV, index=False)

    # Summary stats
    total = len(classified_df)
    trees_only = classified_df[classified_df["veg_type"] == "tree"]
    shrubs_only = classified_df[classified_df["veg_type"] == "shrub"]

    summary = f"""
================================================================================
  CLASSIFICATION SUMMARY
================================================================================

  Total crowns classified: {total:,}

  TREES (>5m):
    Total:     {len(trees_only):,}
    Deciduous: {(trees_only['leaf_type']=='deciduous').sum():,} ({100*(trees_only['leaf_type']=='deciduous').sum()/len(trees_only):.1f}%)
    Conifer:   {(trees_only['leaf_type']=='conifer').sum():,} ({100*(trees_only['leaf_type']=='conifer').sum()/len(trees_only):.1f}%)

  SHRUBS (2-5m):
    Total:     {len(shrubs_only):,}
    Deciduous: {(shrubs_only['leaf_type']=='deciduous').sum():,} ({100*(shrubs_only['leaf_type']=='deciduous').sum()/len(shrubs_only):.1f}%)
    Conifer:   {(shrubs_only['leaf_type']=='conifer').sum():,} ({100*(shrubs_only['leaf_type']=='conifer').sum()/len(shrubs_only):.1f}%)

================================================================================
"""
    print(summary)

    with open(MODEL_REPORT, "a") as f:
        f.write(summary)

    log.info("Done!")


if __name__ == "__main__":
    main()
