# tuning.py
import optuna
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from modeling import setup_logger, config

def optimize_xgb(X_train, y_train, sample_weights):
    """Uses Optuna for XGBoost Bayesian optimization to maximize CV ROC-AUC."""
    logger = setup_logger()
    logger.info("Starting XGBoost Optuna Bayesian Optimization ...")
    tscv = TimeSeriesSplit(n_splits=3, gap=1)
    
    def objective(trial):
        # Suggest hyperparameters for XGBoost
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'tree_method': 'hist', 'random_state': 2026, 'eval_metric': 'logloss'
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    logger.info(f"XGB Best Trial ROC-AUC: {study.best_value:.4f}")
    return study.best_params

def optimize_lgbm(X_train, y_train, sample_weights):
    """Uses Optuna for LightGBM Bayesian optimization to maximize CV ROC-AUC."""
    logger = setup_logger()
    logger.info("Starting LightGBM Optuna Bayesian Optimization ...")
    tscv = TimeSeriesSplit(n_splits=3, gap=1)
    
    def objective(trial):
        # Suggest hyperparameters for LightGBM
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 2026, 'verbose': -1
        }
        model = LGBMClassifier(**param)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[train_idx]
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = model.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, preds))
        return np.mean(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    logger.info(f"LGBM Best Trial ROC-AUC: {study.best_value:.4f}")
    return study.best_params

def optimize_cat(X_train, y_train, sample_weights):
    """Uses Optuna for CatBoost Bayesian optimization to maximize CV ROC-AUC."""
    logger = setup_logger()
    logger.info("Starting CatBoost Optuna Bayesian Optimization ...")
    tscv = TimeSeriesSplit(n_splits=3, gap=1)
    
    def objective(trial):
        # Suggest hyperparameters for CatBoost
        param = {
            'iterations': trial.suggest_int('iterations', 100, 500, step=50),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_seed': 2026, 'verbose': False
        }
        model = CatBoostClassifier(**param)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[train_idx]
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = model.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, preds))
        return np.mean(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    logger.info(f"CatBoost Best Trial ROC-AUC: {study.best_value:.4f}")
    return study.best_params
