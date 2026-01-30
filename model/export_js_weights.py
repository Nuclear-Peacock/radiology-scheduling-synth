# model/export_js_weights.py
"""
Educational Use Only (Non-Clinical)
Export trained PyTorch MLP weights to JSON for pure-JS inference in the browser.

Outputs:
- docs/models/duration_mlp.json
- docs/models/noshow_mlp.json
"""

from __future__ import annotations
import os, json
import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def linear_to_json(linear: nn.Linear):
    W = linear.weight.detach().cpu().numpy().tolist()  # [out][in]
    b = linear.bias.detach().cpu().numpy().tolist()    # [out]
    return {"W": W, "b": b}


def export_mlp_json(model: nn.Module, out_path: str):
    # Extract Linear layers in order
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    payload = {
        "layers": [linear_to_json(l) for l in linears],
        "activation": "relu",
        "note": "Educational only. Synthetic model."
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    art_dir = os.path.join(repo_root, "model_artifacts")
    docs_models = os.path.join(repo_root, "docs", "models")
    os.makedirs(docs_models, exist_ok=True)

    # Load feature dims
    import json as _json
    with open(os.path.join(art_dir, "duration_features.json"), "r", encoding="utf-8") as f:
        dur_meta = _json.load(f)
    with open(os.path.join(art_dir, "noshow_features.json"), "r", encoding="utf-8") as f:
        ns_meta = _json.load(f)

    dur_in = len(dur_meta["feature_columns"])
    ns_in = len(ns_meta["feature_columns"])

    # Duration
    dur = MLPRegressor(dur_in)
    dur.load_state_dict(torch.load(os.path.join(art_dir, "duration_model.pt"), map_location="cpu"))
    dur.eval()
    export_mlp_json(dur, os.path.join(docs_models, "duration_mlp.json"))

    # No-show
    ns = MLPClassifier(ns_in)
    ns.load_state_dict(torch.load(os.path.join(art_dir, "noshow_model.pt"), map_location="cpu"))
    ns.eval()
    export_mlp_json(ns, os.path.join(docs_models, "noshow_mlp.json"))

    # Copy feature metadata too (for later true feature encoding)
    for name in ["duration_features.json", "noshow_features.json"]:
        src = os.path.join(art_dir, name)
        dst = os.path.join(docs_models, name)
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)

    print("Exported JS-weight models to docs/models/")

if __name__ == "__main__":
    main()
