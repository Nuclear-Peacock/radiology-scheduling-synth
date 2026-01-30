# model/export_js_weights.py
"""
Educational Use Only (Non-Clinical)
Export trained PyTorch MLP weights to JSON for pure-JS inference in the browser.

Inputs (produced by training workflow):
- model_artifacts/duration_model.pt
- model_artifacts/noshow_model.pt
- model_artifacts/duration_features.json
- model_artifacts/noshow_features.json

Outputs:
- docs/models/duration_mlp.json
- docs/models/noshow_mlp.json
- docs/models/duration_features.json
- docs/models/noshow_features.json
"""

from __future__ import annotations

import os
import json
import shutil
import torch
from torch import nn


# IMPORTANT: Must match attribute names used in your training scripts: self.net = nn.Sequential(...)
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def _linear_to_json(layer: nn.Linear) -> dict:
    # W: [out][in], b: [out]
    return {
        "W": layer.weight.detach().cpu().numpy().tolist(),
        "b": layer.bias.detach().cpu().numpy().tolist(),
    }


def _export_mlp_to_json(model: nn.Module, out_path: str) -> None:
    # Extract all Linear layers in execution order (Sequential guarantees this order)
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    payload = {
        "layers": [_linear_to_json(l) for l in linears],
        "activation": "relu",
        "output": "linear",  # regressor output is linear; classifier output is logits (still linear)
        "note": "Educational only. Synthetic model.",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    art_dir = os.path.join(repo_root, "model_artifacts")
    docs_models_dir = os.path.join(repo_root, "docs", "models")
    os.makedirs(docs_models_dir, exist_ok=True)

    # Verify inputs exist
    need = [
        os.path.join(art_dir, "duration_model.pt"),
        os.path.join(art_dir, "noshow_model.pt"),
        os.path.join(art_dir, "duration_features.json"),
        os.path.join(art_dir, "noshow_features.json"),
    ]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    # Load feature metadata (for input dimensions)
    with open(os.path.join(art_dir, "duration_features.json"), "r", encoding="utf-8") as f:
        dur_meta = json.load(f)
    with open(os.path.join(art_dir, "noshow_features.json"), "r", encoding="utf-8") as f:
        ns_meta = json.load(f)

    dur_in = len(dur_meta["feature_columns"])
    ns_in = len(ns_meta["feature_columns"])

    # ----- Duration model -----
    dur = MLPRegressor(dur_in)
    dur_state = torch.load(os.path.join(art_dir, "duration_model.pt"), map_location="cpu")
    dur.load_state_dict(dur_state)
    dur.eval()
    _export_mlp_to_json(dur, os.path.join(docs_models_dir, "duration_mlp.json"))

    # ----- No-show model -----
    ns = MLPClassifier(ns_in)
    ns_state = torch.load(os.path.join(art_dir, "noshow_model.pt"), map_location="cpu")
    ns.load_state_dict(ns_state)
    ns.eval()
    _export_mlp_to_json(ns, os.path.join(docs_models_dir, "noshow_mlp.json"))

    # Copy preprocessing metadata alongside weights
    shutil.copyfile(
        os.path.join(art_dir, "duration_features.json"),
        os.path.join(docs_models_dir, "duration_features.json"),
    )
    shutil.copyfile(
        os.path.join(art_dir, "noshow_features.json"),
        os.path.join(docs_models_dir, "noshow_features.json"),
    )

    print("Exported:")
    print("- docs/models/duration_mlp.json")
    print("- docs/models/noshow_mlp.json")
    print("- docs/models/duration_features.json")
    print("- docs/models/noshow_features.json")
    print("\nEducational Use Only (Non-Clinical).")


if __name__ == "__main__":
    main()
