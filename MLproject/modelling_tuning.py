"""
modelling_tuning.py  ─  Hyperparameter Tuning Corn Yield Prediction
=====================================================================
Level      : Advanced
Logging    : FULL MANUAL (mlflow.log_param, mlflow.log_metric, mlflow.log_artifact)
             TANPA mlflow.sklearn.autolog()
Metrik     : rmse, mae, r2, mse, mape, training_time_seconds,
             best_cv_rmse, n_features, feature_importance (CSV + artefak)
Tracking   : DagsHub MLflow Tracking Server
Dataset    : preprocessed_corn_data.csv

Cara pakai:
  python modelling_tuning.py --strategy random --n_iter 30
  python modelling_tuning.py --strategy grid
"""

import argparse
import logging
import os
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, train_test_split,
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("corn-tuning")

# ── Konstanta ─────────────────────────────────────────────────────────────────
DROP_COLS  = ["County", "Farmer", "Crop", "Power source",
              "Water source", "Crop insurance", "Fertilizer_Category"]
TARGET_COL = "Yield"

PARAM_GRID = {
    "n_estimators"     : [50, 100, 200],
    "max_depth"        : [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features"     : ["sqrt", "log2"],
}

PARAM_DIST = {
    "n_estimators"     : [50, 100, 150, 200, 300],
    "max_depth"        : [5, 8, 10, 12, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features"     : ["sqrt", "log2"],
    "bootstrap"        : [True, False],
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(data_path: str):
    logger.info(f"Memuat dataset: {data_path}")
    df = pd.read_csv(data_path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    logger.info(f"Shape X: {X.shape} | y: {y.shape}")
    return X, y


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae    = float(mean_absolute_error(y_test, y_pred))
    mse    = float(mean_squared_error(y_test, y_pred))
    r2     = float(r2_score(y_test, y_pred))
    # MAPE — hindari division by zero
    mask   = y_test != 0
    mape   = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100) if mask.any() else 0.0
    return {"rmse": rmse, "mae": mae, "mse": mse, "r2": r2, "mape": mape}


def save_feature_importance(model, feature_names) -> str:
    fi = pd.DataFrame({
        "feature"   : feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    path = "feature_importance.csv"
    fi.to_csv(path, index=False)
    logger.info(f"Top-5 features:\n{fi.head().to_string()}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def run_tuning(
    data_path      : str,
    strategy       : str,
    n_iter         : int,
    cv_folds       : int,
    test_size      : float,
    random_state   : int,
    experiment_name: str,
):
    # ── Setup DagsHub MLflow Tracking ─────────────────────────────────────────
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    mlflow.set_experiment(experiment_name)

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X_train.shape[1]}")

    # ── Searcher ──────────────────────────────────────────────────────────────
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    base_model = RandomForestRegressor(random_state=random_state)

    if strategy == "grid":
        searcher = GridSearchCV(
            estimator=base_model, param_grid=PARAM_GRID,
            cv=cv, scoring="neg_root_mean_squared_error",
            n_jobs=-1, verbose=1, refit=True,
        )
        run_name = "Tuning_GridSearch_CornYield"
    else:
        searcher = RandomizedSearchCV(
            estimator=base_model, param_distributions=PARAM_DIST,
            n_iter=n_iter, cv=cv, scoring="neg_root_mean_squared_error",
            n_jobs=-1, verbose=1, random_state=random_state, refit=True,
        )
        run_name = f"Tuning_RandomSearch_{n_iter}iter_CornYield"

    # ── MLflow Run — FULL MANUAL LOGGING (no autolog) ─────────────────────────
    with mlflow.start_run(run_name=run_name) as run:

        # -- Log parameter tuning & data --
        mlflow.log_param("strategy",      strategy)
        mlflow.log_param("cv_folds",      cv_folds)
        mlflow.log_param("test_size",     test_size)
        mlflow.log_param("random_state",  random_state)
        mlflow.log_param("n_iter",        n_iter if strategy == "random" else "exhaustive")
        mlflow.log_param("data_path",     os.path.basename(data_path))
        mlflow.log_param("n_train",       len(X_train))
        mlflow.log_param("n_test",        len(X_test))
        mlflow.log_param("n_features",    X_train.shape[1])
        mlflow.log_param("model_type",    "RandomForestRegressor")

        # -- Training + timing --
        logger.info("Memulai tuning...")
        t_start = time.time()
        searcher.fit(X_train, y_train)
        training_time = round(time.time() - t_start, 2)

        best_model   = searcher.best_estimator_
        best_params  = searcher.best_params_
        best_cv_rmse = float(-searcher.best_score_)

        # -- Log best hyperparameter (manual) --
        mlflow.log_param("best_n_estimators",      best_params.get("n_estimators"))
        mlflow.log_param("best_max_depth",         best_params.get("max_depth"))
        mlflow.log_param("best_min_samples_split", best_params.get("min_samples_split"))
        mlflow.log_param("best_min_samples_leaf",  best_params.get("min_samples_leaf"))
        mlflow.log_param("best_max_features",      best_params.get("max_features"))
        mlflow.log_param("best_bootstrap",         best_params.get("bootstrap", True))

        # -- Compute & log semua metrik (manual) --
        metrics = compute_metrics(best_model, X_test, y_test)

        # Metrik utama
        mlflow.log_metric("test_rmse",          metrics["rmse"])
        mlflow.log_metric("test_mae",           metrics["mae"])
        mlflow.log_metric("test_mse",           metrics["mse"])
        mlflow.log_metric("test_r2",            metrics["r2"])
        mlflow.log_metric("test_mape",          metrics["mape"])

        # Metrik CV
        mlflow.log_metric("best_cv_rmse",       best_cv_rmse)

        # Metrik tambahan (Advanced)
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("n_cv_candidates",
                          len(searcher.cv_results_["mean_test_score"]))

        # Training metrics
        y_train_pred = best_model.predict(X_train)
        train_rmse   = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        train_r2     = float(r2_score(y_train, y_train_pred))
        mlflow.log_metric("train_rmse",         train_rmse)
        mlflow.log_metric("train_r2",           train_r2)
        mlflow.log_metric("overfit_gap_rmse",   round(train_rmse - metrics["rmse"], 4))

        # -- Log model (manual) --
        mlflow.sklearn.log_model(
            sk_model       = best_model,
            artifact_path  = "random_forest_model",
            input_example  = X_train.iloc[:3],
        )

        # -- Log feature importance CSV sebagai artefak --
        fi_path = save_feature_importance(best_model, X_train.columns)
        mlflow.log_artifact(fi_path)

        # -- Print ringkasan --
        print("\n" + "═" * 60)
        print(f"  Run ID            : {run.info.run_id}")
        print(f"  Strategy          : {strategy.upper()}")
        print(f"  Training time     : {training_time}s")
        print(f"  Best params       : {best_params}")
        print(f"  Best CV RMSE      : {best_cv_rmse:.4f}")
        print(f"  Test RMSE         : {metrics['rmse']:.4f}")
        print(f"  Test MAE          : {metrics['mae']:.4f}")
        print(f"  Test MSE          : {metrics['mse']:.4f}")
        print(f"  Test R²           : {metrics['r2']:.4f}")
        print(f"  Test MAPE         : {metrics['mape']:.2f}%")
        print(f"  Train RMSE        : {train_rmse:.4f}")
        print(f"  Train R²          : {train_r2:.4f}")
        print(f"  Overfit gap RMSE  : {round(train_rmse - metrics['rmse'], 4):.4f}")
        print("═" * 60 + "\n")

        logger.info("Logging selesai. Semua metrik & artefak tersimpan di MLflow.")

    return best_model, best_params, metrics


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning Corn Yield - Manual MLflow Logging")
    parser.add_argument("--data_path",       type=str,   default="preprocessed_corn_data.csv")
    parser.add_argument("--strategy",        type=str,   default="random",
                        choices=["grid", "random"])
    parser.add_argument("--n_iter",          type=int,   default=30)
    parser.add_argument("--cv_folds",        type=int,   default=5)
    parser.add_argument("--test_size",       type=float, default=0.2)
    parser.add_argument("--random_state",    type=int,   default=42)
    parser.add_argument("--experiment_name", type=str,
                        default="CornYield_HyperparamTuning")
    args = parser.parse_args()

    run_tuning(
        data_path       = args.data_path,
        strategy        = args.strategy,
        n_iter          = args.n_iter,
        cv_folds        = args.cv_folds,
        test_size       = args.test_size,
        random_state    = args.random_state,
        experiment_name = args.experiment_name,
    )
