from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data import load_config, load_dataset, set_global_seed, split_data


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def build_preprocessor(X_train: pd.DataFrame, config: dict) -> ColumnTransformer:
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
                        ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )


def build_models(config: dict) -> dict:
    seed = int(config["seed"])
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=int(config["models"]["logistic_regression"]["max_iter"]),
            random_state=seed,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=int(config["models"]["knn"]["n_neighbors"])
        ),
        "SVM": SVC(
            C=float(config["models"]["svm"]["C"]),
            kernel=config["models"]["svm"]["kernel"],
            gamma=config["models"]["svm"]["gamma"],
            probability=True,
            random_state=seed,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=int(config["models"]["decision_tree"]["max_depth"]),
            min_samples_leaf=int(config["models"]["decision_tree"]["min_samples_leaf"]),
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=int(config["models"]["random_forest"]["n_estimators"]),
            min_samples_leaf=int(config["models"]["random_forest"]["min_samples_leaf"]),
            random_state=seed,
            n_jobs=-1,
        ),
    }


def build_drift_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    numeric = X_train.select_dtypes(include=np.number)
    stats = {
        col: {
            "mean": float(numeric[col].mean()),
            "std": float(numeric[col].std(ddof=0)) if float(numeric[col].std(ddof=0)) > 0 else 1.0,
        }
        for col in numeric.columns
    }
    return {
        "numeric_feature_stats": stats,
        "training_positive_rate": float(y_train.mean()),
        "training_sample_size": int(len(y_train)),
    }


def main() -> None:
    config = load_config()
    set_global_seed(int(config["seed"]))

    df = load_dataset(config)
    X_train, X_test, y_train, y_test = split_data(df, config)

    preprocessor = build_preprocessor(X_train, config)
    models = build_models(config)

    cv = StratifiedKFold(
        n_splits=int(config["cv"]["n_splits"]),
        shuffle=True,
        random_state=int(config["seed"]),
    )
    scoring = {"roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1"}

    cv_rows = []
    trained = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
                "cv_precision_mean": float(np.mean(scores["test_precision"])),
                "cv_recall_mean": float(np.mean(scores["test_recall"])),
                "cv_f1_mean": float(np.mean(scores["test_f1"])),
            }
        )
        pipe.fit(X_train, y_train)
        trained[name] = pipe

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_roc_auc_mean", ascending=False)
    best_model_name = cv_df.iloc[0]["model"]
    best_pipeline = trained[best_model_name]

    probs = best_pipeline.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    target_precision = float(config["business"]["target_precision"])

    candidates = [i for i, p in enumerate(precisions[:-1]) if p >= target_precision]
    if candidates:
        idx = max(candidates, key=lambda i: recalls[i])
    else:
        idx = int(np.argmax(precisions[:-1]))

    threshold = float(thresholds[idx])
    preds = (probs >= threshold).astype(int)

    out_dir = Path(config["artifacts"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / config["artifacts"]["model_file"]
    threshold_path = out_dir / config["artifacts"]["threshold_file"]
    metrics_path = out_dir / config["artifacts"]["metrics_file"]
    lineage_path = out_dir / config["artifacts"].get("lineage_file", "lineage.json")

    joblib.dump(best_pipeline, model_path)
    threshold_path.write_text(str(threshold), encoding="utf-8")

    run_id = str(uuid.uuid4())
    config_hash = _sha256_file(Path("config.yaml"))
    dataset_hash = _sha256_file(Path(config["data"]["path"]))
    model_hash = _sha256_file(model_path)

    metrics = {
        "run_id": run_id,
        "best_model_name": best_model_name,
        "calibration": {"type": "threshold", "target_precision": target_precision, "threshold": threshold},
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "cv_ranking": cv_df.to_dict(orient="records"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lineage = {
        "run_id": run_id,
        "dataset": {
            "path": config["data"]["path"],
            "sha256": dataset_hash,
        },
        "config": {
            "path": "config.yaml",
            "sha256": config_hash,
        },
        "model": {
            "path": str(model_path),
            "sha256": model_hash,
        },
        "threshold": {
            "path": str(threshold_path),
            "sha256": _sha256_file(threshold_path),
        },
    }
    lineage_path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")

    baseline = build_drift_baseline(X_train, y_train)
    baseline_path = out_dir / config["artifacts"].get("drift_baseline_file", "drift_baseline.json")
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
