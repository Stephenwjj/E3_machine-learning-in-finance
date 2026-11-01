import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Executes the Filter and Embedded feature selection funnel
def feature_selection_funnel(X: pd.DataFrame, y: pd.Series, mi_top_k: int, corr_thresh: float, l1_c: float, logger) -> list:

    logger.info(f"Starting feature selection. Initial features: {X.shape[1]}")
    
    # 1. Filter Method: Mutual Information
    mi = mutual_info_classif(X, y, random_state=42)
    mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    keep_mi = mi_s.head(mi_top_k).index.tolist()
    logger.info(f"Filter (MI) retained {len(keep_mi)} features.")
    
    # 2. Filter Method: Correlation Pruning
    corr = X[keep_mi].corr().abs()
    drop_cols = set()
    for i in range(len(keep_mi)):
        for j in range(i+1, len(keep_mi)):
            if corr.iloc[i, j] > corr_thresh:
                a, b = keep_mi[i], keep_mi[j]
                drop_cols.add(a if mi_s[a] < mi_s[b] else b)
    keep_corr = [c for c in keep_mi if c not in drop_cols]
    logger.info(f"Filter (Correlation) retained {len(keep_corr)} features.")
    
    # 3. Embedded Method: L1 Regularization (Lasso)
    X_scaled = StandardScaler().fit_transform(X[keep_corr])
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=l1_c, class_weight='balanced', max_iter=2000)
    l1_model.fit(X_scaled, y)
    
    # Keep only features where the L1 coefficient is not shrunk to zero
    final_feats = [c for c, w in zip(keep_corr, l1_model.coef_[0]) if abs(w) > 1e-5]
    logger.info(f"Embedded (L1) retained {len(final_feats)} final features.")
    
    return final_feats