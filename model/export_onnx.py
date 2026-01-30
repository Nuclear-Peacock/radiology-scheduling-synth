# model/export_onnx.py
"""
Educational Use Only (Non-Clinical)
Exports trained PyTorch models to ONNX for browser inference later.

Outputs:
- ui/public/models/duration.onnx
- ui/public/models/noshow.onnx
- ui/public/models/duration_features.json
- ui/public/models/noshow_features.json
"""

from __future__ import annotations
import os, shutil
import json
import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    art_dir = os.path.join(repo_root, "model_artifacts")
    ui_models = os.path.join(repo_root, "ui", "public", "models")
    os.makedirs(ui_models, exist_ok=True)

    # Copy preprocessing JSONs into UI
    for name in ["duration_features.json", "noshow_features.json"]:
        src = os.path.join(art_dir, name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing {src}. Train models first.")
        shutil.copyfile(src, os.path.join(ui_models, name))

    # Load feature dims
    with open(os.path.join(art_dir, "duration_features.json"), "r", encoding="utf-8") as f:
        dur_meta = json.load(f)
    with open(os.path.join(art_dir, "noshow_features.json"), "r", encoding="utf-8") as f:
        ns_meta = json.load(f)

    dur_in = len(dur_meta["feature_columns"])
    ns_in = len(ns_meta["feature_columns"])

    # Export duration
    dur = MLPRegressor(dur_in)
    dur.load_state_dict(torch.load(os.path.join(art_dir, "duration_model.pt"), map_location="cpu"))
    dur.eval()

    dummy = torch.randn(1, dur_in)
    torch.onnx.export(
        dur,
        dummy,
        os.path.join(ui_models, "duration.onnx"),
        input_names=["x"],
        output_names=["duration_min"],
        opset_version=17,
        dynamic_axes={"x": {0: "batch"}}
    )

    # Export no-show
    ns = MLPClassifier(ns_in)
    ns.load_state_dict(torch.load(os.path.join(art_dir, "noshow_model.pt"), map_location="cpu"))
    ns.eval()

    dummy2 = torch.randn(1, ns_in)
    torch.onnx.export(
        ns,
        dummy2,
        os.path.join(ui_models, "noshow.onnx"),
        input_names=["x"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"x": {0: "batch"}}
    )

    print("Wrote ONNX + feature metadata to ui/public/models/")

if __name__ == "__main__":
    main()
