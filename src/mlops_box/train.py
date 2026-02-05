from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
import mlflow

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .evaluate import compute_metrics
from .export import export_sklearn
from .predict import predict_proba

console = Console()

# Robust config path: repo_root/configs (works with entry-points)
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def build_model(cfg: DictConfig) -> Pipeline:
    if str(cfg.model.type) != "logreg":
        raise ValueError(f"Unknown model.type={cfg.model.type}")

    clf = LogisticRegression(C=float(cfg.model.C), max_iter=int(cfg.model.max_iter))
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return pipe


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    val_size: float,
    test_size: float,
):
    """
    val_size and test_size are fractions of the FULL dataset.
    train fraction = 1 - (val_size + test_size)
    """
    if not (0.0 < val_size < 1.0) or not (0.0 < test_size < 1.0):
        raise ValueError("val_size and test_size must be in (0,1)")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    rest_size = val_size + test_size

    X_train, X_rest, y_train, y_rest = train_test_split(
        X,
        y,
        test_size=rest_size,
        random_state=seed,
        stratify=y,
    )

    # fraction of REST that should go to val
    val_frac_of_rest = val_size / rest_size

    X_val, X_test, y_val, y_test = train_test_split(
        X_rest,
        y_rest,
        test_size=(1.0 - val_frac_of_rest),
        random_state=seed,
        stratify=y_rest,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    console.print("[bold]Config:[/bold]")
    console.print(OmegaConf.to_yaml(cfg))

    seed = int(cfg.seed)
    set_seed(seed)

    # Synthetic classification data
    X, y = make_classification(
        n_samples=int(cfg.data.n_samples),
        n_features=int(cfg.data.n_features),
        n_informative=int(cfg.data.n_informative),
        n_redundant=int(cfg.data.n_redundant),
        class_sep=float(cfg.data.class_sep),
        flip_y=float(cfg.data.flip_y),
        random_state=seed,
    )

    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X,
        y,
        seed=seed,
        val_size=float(cfg.data.val_size),
        test_size=float(cfg.data.test_size),
    )

    feature_names = [f"f{i:02d}" for i in range(X.shape[1])]

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    # MLflow setup
    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment))

    with mlflow.start_run(run_name=str(cfg.run_name)):
        mlflow.log_params(
            {
                "seed": seed,
                "n_samples": int(cfg.data.n_samples),
                "n_features": int(cfg.data.n_features),
                "val_size": float(cfg.data.val_size),
                "test_size": float(cfg.data.test_size),
                "model": str(cfg.model.type),
                "C": float(cfg.model.C),
                "max_iter": int(cfg.model.max_iter),
            }
        )

        model = build_model(cfg)
        model.fit(X_train, y_train)

        p_val = predict_proba(model, X_val)
        p_test = predict_proba(model, X_test)

        m_val = compute_metrics(y_val, p_val)
        m_test = compute_metrics(y_test, p_test)

        mlflow.log_metrics(
            {
                "val_accuracy": m_val.accuracy,
                "val_f1": m_val.f1,
                "val_auc": m_val.auc,
                "test_accuracy": m_test.accuracy,
                "test_f1": m_test.f1,
                "test_auc": m_test.auc,
            }
        )

        # Export model
        model_dir = Path("models/latest")
        model_path = export_sklearn(model, feature_names, model_dir)

        summary = {
            "out_dir": str(out_dir),
            "model_path": str(model_path),
            "val": m_val.__dict__,
            "test": m_test.__dict__,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Log artifacts
        mlflow.log_artifact(str(out_dir / "config.yaml"))
        mlflow.log_artifact(str(out_dir / "summary.json"))
        mlflow.log_artifact(str(model_dir / "signature.json"))
        mlflow.log_artifact(str(model_dir / "model.joblib"))

        console.print("[green]Done[/green]")
        console.print(f"Saved model: {model_path}")
        console.print(f"Run summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
