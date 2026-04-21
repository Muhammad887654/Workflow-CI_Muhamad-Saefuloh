import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_and_prepare_data(data_path):
    """Load preprocessed CSV and return feature matrix X and target y."""
    df = pd.read_csv(data_path)

    # Drop non-numeric / identifier columns
    drop_cols = ['County', 'Farmer', 'Crop', 'Power source',
                 'Water source', 'Crop insurance', 'Fertilizer_Category']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    target_col = 'Yield'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_model(data_path, n_estimators, max_depth, random_state, test_size):
    X, y = load_and_prepare_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing  samples : {len(X_test)}")
    print(f"Features         : {X_train.shape[1]}")

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    with mlflow.start_run(run_name="RandomForest_CornYield"):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        mlflow.log_metric("rmse_manual", rmse)
        mlflow.log_metric("r2_manual",   r2)

        # Explicitly log the model to MLflow
        mlflow.sklearn.log_model(model, "model")

        print(f"\n── Hasil Evaluasi ──")
        print(f"RMSE    : {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Run ID  : {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest Corn Yield Model")
    parser.add_argument("--data_path",    type=str,   default="preprocessed_corn_data.csv")
    parser.add_argument("--n_estimators", type=int,   default=100)
    parser.add_argument("--max_depth",    type=int,   default=10)
    parser.add_argument("--random_state", type=int,   default=42)
    parser.add_argument("--test_size",    type=float, default=0.2)
    args = parser.parse_args()

    print("Memulai pelatihan model Corn Yield Prediction...")
    train_model(
        data_path    = args.data_path,
        n_estimators = args.n_estimators,
        max_depth    = args.max_depth,
        random_state = args.random_state,
        test_size    = args.test_size,
    )
    print("Pelatihan selesai.")
