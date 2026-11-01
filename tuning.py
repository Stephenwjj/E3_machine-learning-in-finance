import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import numpy as np
import config
from utils import setup_logger

def optimize_xgb(X_train, y_train, sample_weights):
    """Uses Optuna for Bayesian optimization to maximize CV ROC-AUC."""
    logger = setup_logger()
    logger.info("Starting Optuna Bayesian Optimization (Max 50 trials)...")
    
    # use TimeSeriesSplit to prevent data leakage
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    
    def objective(trial):
        # define the hyperparameters search range
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'tree_method': 'hist',
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        model = XGBClassifier(**param)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[train_idx]
            
            model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, preds))
            
        return np.mean(cv_scores)

    # print Optuna result
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    logger.info(f"Best Trial ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best Parameters: {study.best_params}")
    
    return study.best_params