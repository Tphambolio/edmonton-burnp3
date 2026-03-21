#!/usr/bin/env python3
"""
Elm Detector with Neighbourhood Features
==========================================
Adds spatial context: what species are nearby trees?
Elm trees cluster along streets and in parks — spatial autocorrelation helps.
"""
import os, json, logging, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from pyproj import Transformer
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("elm_nb")

BASE = "/home/rpas/Documents/City-of-Edmonton/Urban-Forestry/LiDAR"

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Elm Detector with Neighbourhood Features")
    log.info("=" * 60)

    # Load crown features + tree data
    log.info("Loading data...")
    feat = pd.read_csv(f"{BASE}/tree_results/crown_features.csv")
    trees = pd.read_csv(f"{BASE}/tree_results/trees_classified.csv",
        usecols=['tile','tree_id','centroid_x','centroid_y','max_height',
                 'crown_area_m2','crown_diameter_m','compactness','eccentricity',
                 'veg_type','perimeter_m','mean_height','std_height'])
    merged = trees.merge(feat, on=['tile','tree_id'], how='inner')
    log.info(f"Crowns: {len(merged):,}")

    # Load inventory
    with open(f"{BASE}/edmonton_tree_inventory.json") as f:
        inv_raw = json.load(f)
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3776', always_xy=True)
    inv = []
    for t in inv_raw:
        genus = t.get('genus','')
        lat, lon = t.get('latitude'), t.get('longitude')
        if not lat or not lon or not genus: continue
        x, y = transformer.transform(float(lon), float(lat))
        inv.append({'x': x, 'y': y, 'is_elm': 1 if genus == 'Ulmus' else 0, 'genus': genus})

    # Match inventory to crowns
    log.info("Spatial matching...")
    crown_xy = merged[['centroid_x','centroid_y']].values
    kdtree = cKDTree(crown_xy)
    inv_xy = np.array([[t['x'], t['y']] for t in inv])
    dists, idxs = kdtree.query(inv_xy, k=1)

    labels = {}
    for i, (d, idx) in enumerate(zip(dists, idxs)):
        if d <= 5.0 and idx not in labels:
            labels[idx] = inv[i]['is_elm']
    log.info(f"Matched: {len(labels):,} ({sum(labels.values()):,} elms)")

    # Build neighbourhood features for ALL crowns (not just matched ones)
    # For each crown, look at the K nearest crowns and compute stats
    log.info("Computing neighbourhood features...")
    
    # First, predict elm probability for ALL crowns using a quick model on matched data
    train_idx = list(labels.keys())
    train_df = merged.iloc[train_idx].copy()
    train_df['is_elm'] = [labels[i] for i in train_idx]
    
    exclude = {'tile','tree_id','centroid_x','centroid_y','veg_type','n_points','is_elm'}
    base_features = [c for c in train_df.columns if c not in exclude]
    
    X_base = train_df[base_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_base = train_df['is_elm'].values
    
    # Quick preliminary model to get elm probabilities for all crowns
    clf_prelim = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    clf_prelim.fit(X_base, y_base)
    
    X_all = merged[base_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    elm_probs_all = clf_prelim.predict_proba(X_all)[:, 1]
    
    # Now compute neighbourhood features using KD-tree
    log.info("Building neighbourhood features with K=10, 25, 50 nearest...")
    
    nb_features = np.zeros((len(merged), 15), dtype=np.float32)
    nb_names = []
    
    for ki, K in enumerate([10, 25, 50]):
        log.info(f"  K={K}...")
        dists_k, idxs_k = kdtree.query(crown_xy, k=K+1)  # +1 because first is self
        
        # For each crown, stats of K nearest neighbours
        for i in range(len(merged)):
            neighbours = idxs_k[i, 1:]  # exclude self
            nb_probs = elm_probs_all[neighbours]
            nb_heights = merged.iloc[neighbours]['max_height'].values
            
            col_base = ki * 5
            nb_features[i, col_base] = nb_probs.mean()      # avg elm probability of neighbours
            nb_features[i, col_base+1] = nb_probs.std()      # variation in elm prob
            nb_features[i, col_base+2] = (nb_probs > 0.5).mean()  # fraction of neighbours that look like elm
            nb_features[i, col_base+3] = nb_heights.mean()    # avg height of neighbours
            nb_features[i, col_base+4] = dists_k[i, -1]      # distance to Kth nearest
        
        if ki == 0:
            nb_names = [f'nb{K}_elm_prob_mean', f'nb{K}_elm_prob_std', f'nb{K}_elm_frac',
                       f'nb{K}_height_mean', f'nb{K}_dist_kth']
        else:
            nb_names += [f'nb{K}_elm_prob_mean', f'nb{K}_elm_prob_std', f'nb{K}_elm_frac',
                        f'nb{K}_height_mean', f'nb{K}_dist_kth']
    
    log.info(f"Neighbourhood features computed: {len(nb_names)}")
    
    # Combine base + neighbourhood features for training set
    nb_train = nb_features[train_idx]
    X_combined = np.hstack([X_base, nb_train])
    all_feature_names = base_features + nb_names
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y_base, test_size=0.2, random_state=42, stratify=y_base)
    
    # === Model A: Base features only ===
    clf_base = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=-1,
        num_leaves=127, min_child_samples=10, subsample=0.9, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    clf_base.fit(X_train[:, :len(base_features)], y_train)
    p_base = clf_base.predict(X_val[:, :len(base_features)])
    f1_base = f1_score(y_val, p_base, average='macro')
    
    log.info(f"\n=== BASE (LiDAR only) — Macro F1: {f1_base:.4f} ===")
    print(classification_report(y_val, p_base, target_names=['Non-Elm','Elm'], digits=3))
    
    # === Model B: Base + Neighbourhood ===
    clf_nb = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=-1,
        num_leaves=127, min_child_samples=10, subsample=0.9, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    clf_nb.fit(X_train, y_train)
    p_nb = clf_nb.predict(X_val)
    f1_nb = f1_score(y_val, p_nb, average='macro')
    
    log.info(f"\n=== BASE + NEIGHBOURHOOD — Macro F1: {f1_nb:.4f} ===")
    print(classification_report(y_val, p_nb, target_names=['Non-Elm','Elm'], digits=3))
    
    # Cross-val
    cv = cross_val_score(clf_nb, X_combined, y_base, cv=5, scoring='f1_macro')
    log.info(f"5-fold CV: {cv.mean():.4f} ± {cv.std():.4f}")
    
    # Feature importance — show neighbourhood features
    importances = sorted(zip(all_feature_names, clf_nb.feature_importances_), key=lambda x: -x[1])
    log.info("\nTop 20 features:")
    for name, imp in importances[:20]:
        marker = " ** NEIGHBOURHOOD" if name.startswith("nb") else ""
        log.info(f"  {name:<30s} {imp:>5.0f}{marker}")
    
    log.info(f"\nImprovement: Base {f1_base:.4f} → +Neighbourhood {f1_nb:.4f} ({(f1_nb-f1_base)*100:+.2f} pp)")
    log.info(f"Done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
