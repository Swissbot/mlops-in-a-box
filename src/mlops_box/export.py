from __future__ import annotations
from pathlib import Path
import json
import joblib

def export_sklearn(model, feature_names: list[str], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    sig = {
        "inputs": [{"name": n, "dtype": "float"} for n in feature_names],
        "outputs": [{"name": "proba", "dtype": "float"}],
    }
    (out_dir / "signature.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    return model_path
