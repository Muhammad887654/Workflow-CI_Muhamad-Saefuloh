"""
modelling_tuning.py  ─  Hyperparameter Tuning Corn Yield Prediction
=====================================================================
Strategi tuning : GridSearchCV (exhaustive) + RandomizedSearchCV (cepat)
Model            : RandomForestRegressor
Logging          : mlflow.sklearn.autolog() — params, metrics, artefak otomatis
Dataset          : preprocessed_corn_data.csv (hasil preprocessing)

Cara pakai
----------
# Tuning penuh (GridSearch):
python modelling_tuning.py --strategy grid --data_path preprocessed_corn_data.csv

# Tuning cepat (RandomizedSearch, cocok CI):
python modelling_tuning.py --strategy random --n_iter 20 --data_path preprocessed_corn_data.csv
"""

import argparse
import logging
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("corn-tuning")

# ── Kolom yang tidak dipakai sebagai fitur ────────────────────────────────────
DROP_COLS   = ["County", "Farmer", "Crop", "Power source",
               "Water source", "Crop insurance", "Fertilizer_Category"]
TARGET_COL  = "Yield"

# ── Search space ──────────────────────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators" : [50, 100, 200],
    "max_depth"    : [5, 10, 15, None],
    "min_samples_split" : [2, 5, 10],
    "min_samples_leaf"  : [1, 2, 4],
    "max_features"      : ["sqrt", "log2"],
}

PARAM_DIST = {
    "n_estimators"      : [50, 100, 150, 200, 300],
    "max_depth"         : [5, 8, 10, 12, 15, None],
    "min_samples_split" : [2, 5, 10],
    "min_samples_leaf"  : [1, 2, 4],
    "max_features"      : ["sqrt", "log2"],
    "bootstrap"         : [True, False],
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(data_path: str):
    logger.info(f"Memuat dataset: {data_path}")
    df = pd.read_csv(data_path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info(f"Shape  ─ X: {X.shape}, y: {y.shape}")
    logger.info(f"Target ─ min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
    return X, y


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "rmse" : float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae"  : float(mean_absolute_error(y_test, y_pred)),
        "r2"   : float(r2_score(y_test, y_pred)),
    }


# ── Main training + tuning ────────────────────────────────────────────────────
def run_tuning(
    data_path   : str,
    strategy    : str,
    n_iter      : int,
    cv_folds    : int,
    test_size   : float,
    random_state: int,
    experiment_name: str,
):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    base_model = RandomForestRegressor(random_state=random_state)

    # Pilih searcher
    if strategy == "grid":
        logger.info("Strategi: GridSearchCV (exhaustive)")
        searcher = GridSearchCV(
            estimator  = base_model,
            param_grid = PARAM_GRID,
            cv         = cv,
            scoring    = "neg_root_mean_squared_error",
            n_jobs     = -1,
            verbose    = 1,
            refit      = True,
        )
        run_name = "Tuning_GridSearch_CornYield"
    else:
        logger.info(f"Strategi: RandomizedSearchCV (n_iter={n_iter})")
        searcher = RandomizedSearchCV(
            estimator          = base_model,
            param_distributions= PARAM_DIST,
            n_iter             = n_iter,
            cv                 = cv,
            scoring            = "neg_root_mean_squared_error",
            n_jobs             = -1,
            verbose            = 1,
            random_state       = random_state,
            refit              = True,
        )
        run_name = f"Tuning_RandomSearch_{n_iter}iter_CornYield"

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # ── autolog aktif: params, metrics, model, feature importance tercatat otomatis ──
    mlflow.sklearn.autolog(
        log_model_signatures=True,
        log_input_examples=True,
        max_tuning_runs=5,          # simpan top-5 child runs dari CV
    )

    with mlflow.start_run(run_name=run_name) as run:
        logger.info("Memulai tuning …")
        searcher.fit(X_train, y_train)

        best_model  = searcher.best_estimator_
        best_params = searcher.best_params_
        best_cv_rmse = -searcher.best_score_

        logger.info(f"Best params  : {best_params}")
        logger.info(f"Best CV RMSE : {best_cv_rmse:.4f}")

        # Evaluasi pada test set
        metrics = evaluate(best_model, X_test, y_test)
        logger.info(f"Test RMSE : {metrics['rmse']:.4f}")
        logger.info(f"Test MAE  : {metrics['mae']:.4f}")
        logger.info(f"Test R²   : {metrics['r2']:.4f}")

        # Log tambahan (melengkapi autolog)
        mlflow.log_param("strategy",    strategy)
        mlflow.log_param("cv_folds",    cv_folds)
        mlflow.log_param("test_size",   test_size)
        mlflow.log_param("data_path",   os.path.basename(data_path))
        mlflow.log_param("n_features",  X_train.shape[1])
        mlflow.log_param("n_train",     len(X_train))

        mlflow.log_metric("best_cv_rmse", best_cv_rmse)
        mlflow.log_metric("test_rmse",    metrics["rmse"])
        mlflow.log_metric("test_mae",     metrics["mae"])
        mlflow.log_metric("test_r2",      metrics["r2"])

        # Simpan feature importance sebagai CSV artefak
        fi = pd.DataFrame({
            "feature"   : X_train.columns,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi_path = "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)
        logger.info(f"Top-5 features:\n{fi.head()}")

        print("\n" + "═" * 55)
        print(f"  Run ID      : {run.info.run_id}")
        print(f"  Strategy    : {strategy.upper()}")
        print(f"  Best Params : {best_params}")
        print(f"  CV RMSE     : {best_cv_rmse:.4f}")
        print(f"  Test RMSE   : {metrics['rmse']:.4f}")
        print(f"  Test MAE    : {metrics['mae']:.4f}")
        print(f"  Test R²     : {metrics['r2']:.4f}")
        print("═" * 55 + "\n")

    return best_model, best_params, metrics


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning ─ Corn Yield Prediction"
    )
    parser.add_argument("--data_path",       type=str,   default="preprocessed_corn_data.csv")
    parser.add_argument("--strategy",        type=str,   default="random",
                        choices=["grid", "random"],
                        help="'grid' = GridSearchCV | 'random' = RandomizedSearchCV")
    parser.add_argument("--n_iter",          type=int,   default=30,
                        help="Jumlah kombinasi (hanya untuk --strategy random)")
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
