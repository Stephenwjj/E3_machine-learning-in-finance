import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

# Executes the Filter and Embedded feature selection funnel
def feature_selection_funnel(X: pd.DataFrame, y: pd.Series, mi_top_k: int, corr_thresh: float, l1_c: float, logger) -> list:

    logger.info(f"Starting feature selection. Initial features: {X.shape[1]}")
    
    # 1. Filter Method: Mutual Information
    mi = mutual_info_classif(X, y, random_state=2026)
    mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    keep_mi = mi_s.head(mi_top_k).index.tolist()
    logger.info(f"Filter (MI) retained {len(keep_mi)} features.")
    
    # 2. Filter Method: Correlation Pruning
    X_mi = X[keep_mi]
    corr_matrix = X_mi.corr().abs()
    
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    drop_cols = set()

    for col in upper.columns:
        high_corr_feats = upper.index[upper[col] > corr_thresh].tolist()
        for feat in high_corr_feats:
            if mi_s[col] > mi_s[feat]:
                drop_cols.add(feat)
            else:
                drop_cols.add(col)
    keep_corr = [c for c in keep_mi if c not in drop_cols]
    logger.info(f"Filter (Correlation) retained {len(keep_corr)} features.")

    # 3. Wrapper Method: RFECV
    outlier_features = ['macd_hist']
    outlier_cols = [c for c in keep_corr if c in outlier_features]
    other_cols = [c for c in keep_corr if c not in outlier_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('robust', RobustScaler(), outlier_cols),
            ('standard', StandardScaler(), other_cols)
        ]
    )
    
    tscv = TimeSeriesSplit(n_splits=3, gap=1)
    est = Pipeline([
        ("sc", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear"))
    ])
    
    rfecv = RFECV(
        estimator=est,
        step=1,
        cv=tscv,
        scoring="roc_auc",
        min_features_to_select=8,
        importance_getter='named_steps.clf.coef_',
        n_jobs=-1
    )
    rfecv.fit(X[keep_corr], y)
    keep_wrap = X[keep_corr].columns[rfecv.support_].tolist()
    logger.info(f"Wrapper (RFECV) retained {len(keep_wrap)} features.")
    
    # 4. Embedded Method: L1 Regularization (Lasso)
    X_scaled = StandardScaler().fit_transform(X[keep_wrap])
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=l1_c, class_weight='balanced', max_iter=2000)
    l1_model.fit(X_scaled, y)
    
    # Keep only features where the coefficient is not shrunk to zero
    final_feats = [c for c, w in zip(keep_wrap, l1_model.coef_[0]) if abs(w) > 1e-6]
    logger.info(f"Embedded (L1) retained {len(final_feats)} final features.")
    logger.info(f"Final selected features: {final_feats}")
    
    return final_feats
