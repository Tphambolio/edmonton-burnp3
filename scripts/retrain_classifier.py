#!/usr/bin/env python3
"""
Retrain Conifer/Deciduous Classifier with Rich LiDAR Features
===============================================================
Uses extracted per-crown features (RGB, intensity, vertical structure,
return ratios) combined with the tree inventory training labels.
"""

import os
import logging
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import json
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retrain")

BASE_DIR = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"
FEATURES_CSV = os.path.join(BASE_DIR, "tree_results/crown_features.csv")
TREES_CSV = os.path.join(BASE_DIR, "tree_results/trees_classified.csv")
INVENTORY_FILE = os.path.join(BASE_DIR, "edmonton_tree_inventory.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "tree_results/trees_classified_v2.csv")
MODEL_FILE = os.path.join(BASE_DIR, "tree_results/classifier_v2.joblib")
REPORT_FILE = os.path.join(BASE_DIR, "tree_results/classification_report_v2.txt")

MATCH_RADIUS = 5.0

CONIFER_GENERA = {
    "Picea", "Pinus", "Larix", "Pseudotsuga", "Thuja",
    "Juniperus", "Abies", "Tsuga", "Taxus",
}

DECIDUOUS_GENERA = {
    "Ulmus", "Fraxinus", "Populus", "Malus", "Quercus", "Acer",
    "Tilia", "Prunus", "Syringa", "Sorbus", "Salix", "Betula",
    "Aesculus", "Crataegus", "Eleagnus", "Pyrus", "Cornus",
    "Amelanchier", "Caragana", "Juglans", "Sambucus",
}

# Feature columns to use (will be filtered to what's available)
FEATURE_COLS = [
    # Original shape features (from trees_classified.csv)
    "max_height", "mean_height", "std_height", "crown_area_m2",
    "crown_diameter_m", "compactness", "eccentricity", "perimeter_m",
    # Height structure
    "h_std", "h_median", "h_p25", "h_p75", "h_p95",
    "h_skew", "h_kurtosis", "h_cv", "h_iqr",
    "pct_lower_quarter", "pct_mid_lower", "pct_mid_upper", "pct_upper_quarter",
    # RGB
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "grvi", "ngrdi", "g_ratio", "r_ratio", "b_ratio",
    "brightness", "rgb_range",
    # Intensity
    "int_mean", "int_std", "int_cv",
    # Return structure
    "pct_first_return", "pct_single_return", "pct_multi_return",
    "mean_num_returns", "penetration_ratio",
    # Classification mix
    "pct_class3", "pct_class4", "pct_class5",
]


def load_inventory():
    """Load and classify tree inventory."""
    with open(INVENTORY_FILE) as f:
        raw = json.load(f)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3776", always_xy=True)
    trees = []
    for t in raw:
        genus = t.get("genus", "")
        lat, lon = t.get("latitude"), t.get("longitude")
        if not lat or not lon or not genus:
            continue
        if genus in CONIFER_GENERA:
            leaf_type = "conifer"
        elif genus in DECIDUOUS_GENERA:
            leaf_type = "deciduous"
        else:
            continue
        x, y = transformer.transform(float(lon), float(lat))
        trees.append({"leaf_type": leaf_type, "genus": genus, "x": x, "y": y})
    return trees


def main():
    log.info("=" * 60)
    log.info("Retrain Classifier with Rich LiDAR Features")
    log.info("=" * 60)

    # Load features
    log.info("Loading crown features...")
    feat_df = pd.read_csv(FEATURES_CSV)
    log.info(f"Features loaded: {len(feat_df):,} crowns, {len(feat_df.columns)} columns")

    # Load original classified trees (for shape features)
    log.info("Loading tree crown data...")
    trees_df = pd.read_csv(TREES_CSV)
    log.info(f"Trees loaded: {len(trees_df):,}")

    # Merge features with tree data
    log.info("Merging features with crown data...")
    merged = trees_df.merge(
        feat_df, on=["tile", "tree_id"], how="inner", suffixes=("", "_feat")
    )
    log.info(f"Merged: {len(merged):,} crowns with features")

    # Load inventory for training labels
    log.info("Loading tree inventory...")
    inventory = load_inventory()
    log.info(f"Inventory: {len(inventory):,} classifiable trees")

    # Spatial match inventory to featured crowns
    log.info("Spatial matching...")
    crown_xy = merged[["centroid_x", "centroid_y"]].values
    tree = cKDTree(crown_xy)

    inv_xy = np.array([[t["x"], t["y"]] for t in inventory])
    distances, indices = tree.query(inv_xy, k=1)

    # Build training set
    matched_indices = set()
    labels = {}
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist <= MATCH_RADIUS and idx not in matched_indices:
            matched_indices.add(idx)
            labels[idx] = inventory[i]["leaf_type"]

    log.info(f"Matched {len(labels):,} unique crowns to inventory")

    # Prepare training data
    train_indices = list(labels.keys())
    train_df = merged.iloc[train_indices].copy()
    train_df["label"] = [labels[i] for i in train_indices]

    # Select available features
    available_features = [c for c in FEATURE_COLS if c in merged.columns]
    log.info(f"Using {len(available_features)} features: {available_features}")

    X_train_full = train_df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train_full = (train_df["label"] == "conifer").astype(int)

    log.info(f"Training set: {len(X_train_full):,} samples")
    log.info(f"  Conifer: {y_train_full.sum():,}, Deciduous: {(~y_train_full.astype(bool)).sum():,}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # Train Random Forest
    log.info("Training Random Forest (v2)...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["deciduous", "conifer"])
    cm = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(clf, X_train_full, y_train_full, cv=5, scoring="accuracy")

    log.info(f"\n{report}")
    log.info(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importance
    importances = sorted(
        zip(available_features, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )

    # ---- Predict all ----
    log.info(f"Predicting for all {len(merged):,} featured crowns...")
    X_all = merged[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    predictions = clf.predict(X_all)
    probabilities = clf.predict_proba(X_all)

    merged["leaf_type_v2"] = np.where(predictions == 1, "conifer", "deciduous")
    merged["conifer_prob_v2"] = probabilities[:, 1]

    # Stats
    conifer_count = (merged["leaf_type_v2"] == "conifer").sum()
    decid_count = (merged["leaf_type_v2"] == "deciduous").sum()

    trees_only = merged[merged["veg_type"] == "tree"]
    shrubs_only = merged[merged["veg_type"] == "shrub"]

    # Save
    log.info(f"Saving to {OUTPUT_CSV}...")
    merged.to_csv(OUTPUT_CSV, index=False)

    # Save model
    joblib.dump(clf, MODEL_FILE)
    log.info(f"Model saved to {MODEL_FILE}")

    # Write report
    report_text = f"""
================================================================================
  CONIFER / DECIDUOUS CLASSIFICATION v2 — WITH RGB + STRUCTURAL FEATURES
================================================================================

  Training Data:    {len(X_train_full):,} inventory-matched crowns
  Train/Test:       {len(X_train):,} / {len(X_test):,}
  Model:            Random Forest (300 trees, max_depth=20)
  Features:         {len(available_features)} LiDAR features (RGB, intensity, structure, returns)

  PERFORMANCE
  -----------
{report}

  Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}

  Confusion Matrix:
                 Pred Deciduous  Pred Conifer
  Act Deciduous    {cm[0][0]:>10,}   {cm[0][1]:>10,}
  Act Conifer      {cm[1][0]:>10,}   {cm[1][1]:>10,}

  FEATURE IMPORTANCE (top 20):
"""
    for fname, imp in importances[:20]:
        bar = "█" * int(imp * 100)
        report_text += f"    {fname:<25s} {imp:.4f}  {bar}\n"

    report_text += f"""
  CITY-WIDE PREDICTIONS ({len(merged):,} crowns with features)
  -------------------------------------------------------
  Deciduous: {decid_count:,} ({100*decid_count/len(merged):.1f}%)
  Conifer:   {conifer_count:,} ({100*conifer_count/len(merged):.1f}%)

  TREES (>5m):
    Deciduous: {(trees_only['leaf_type_v2']=='deciduous').sum():,} ({100*(trees_only['leaf_type_v2']=='deciduous').sum()/len(trees_only):.1f}%)
    Conifer:   {(trees_only['leaf_type_v2']=='conifer').sum():,} ({100*(trees_only['leaf_type_v2']=='conifer').sum()/len(trees_only):.1f}%)

  SHRUBS (2-5m):
    Deciduous: {(shrubs_only['leaf_type_v2']=='deciduous').sum():,} ({100*(shrubs_only['leaf_type_v2']=='deciduous').sum()/len(shrubs_only):.1f}%)
    Conifer:   {(shrubs_only['leaf_type_v2']=='conifer').sum():,} ({100*(shrubs_only['leaf_type_v2']=='conifer').sum()/len(shrubs_only):.1f}%)

  COMPARISON: v1 (shape only) → v2 (shape + RGB + structure)
  -----------------------------------------------------------
  v1 accuracy: 83.3%
  v2 accuracy: {cv_scores.mean()*100:.1f}%
  Improvement: {(cv_scores.mean()-0.833)*100:+.1f} percentage points

================================================================================
"""
    print(report_text)
    with open(REPORT_FILE, "w") as f:
        f.write(report_text)
    log.info(f"Report: {REPORT_FILE}")
    log.info("Done!")


if __name__ == "__main__":
    main()
