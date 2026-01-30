# model/train_duration.py
"""
Educational Use Only (Non-Clinical)
Trains a simple MLP regression model to predict exam duration (minutes).
Exports:
- model_artifacts/duration_model.pt
- model_artifacts/duration_features.json  (feature column order for UI)
"""

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def make_features(df: pd.DataFrame, y_col: str, feature_columns: list[str] | None = None):
    # One-hot categorical + keep numeric/binary
    cat_cols = ["patient_type", "resource_modality", "exam_code", "priority", "scanner_id", "mobility"]
    num_cols = ["requires_contrast", "requires_sedation", "isolation", "hour", "dow", "lead_time_days"]

    # Ensure expected columns exist
    for c in cat_cols + num_cols + [y_col]:
        if c not in df.columns:
            df[c] = 0

    X_cat = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols)
    X_num = df[num_cols].astype(float)

    X = pd.concat([X_num, X_cat], axis=1)
    if feature_columns is None:
        feature_columns = X.columns.tolist()
    else:
        # align to stored columns
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_columns]

    y = df[y_col].astype(float).to_numpy()
    X = X.to_numpy(dtype=np.float32)
    y = y.astype(np.float32)

    # Standardize numeric scale lightly (simple global standardization)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-6] = 1.0
    Xn = (X - X_mean) / X_std

    return Xn, y, feature_columns, X_mean, X_std


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


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(repo_root, "data", "derived", "train_duration.csv")
    out_dir = os.path.join(repo_root, "model_artifacts")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["duration_min"]).copy()
    df["duration_min"] = df["duration_min"].clip(lower=1.0, upper=240.0)

    # Train/val split
    rng = np.random.default_rng(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(0.85 * len(df))
    tr_idx, va_idx = idx[:split], idx[split:]

    Xtr, ytr, feature_cols, X_mean, X_std = make_features(df.iloc[tr_idx], "duration_min")
    Xva, yva, _, _, _ = make_features(df.iloc[va_idx], "duration_min", feature_cols)

    device = "cpu"
    model = MLPRegressor(in_dim=Xtr.shape[1]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=512, shuffle=True
    )

    model.train()
    for epoch in range(10):
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # quick validation MAE
        model.eval()
        with torch.no_grad():
            vpred = model(torch.from_numpy(Xva).to(device)).cpu().numpy()
        mae = float(np.mean(np.abs(vpred - yva)))
        model.train()
        print(f"epoch {epoch+1}/10  train_loss={np.mean(losses):.4f}  val_MAE={mae:.2f} min")

    # Save model + preprocessing
    torch.save(model.state_dict(), os.path.join(out_dir, "duration_model.pt"))

    meta = {
        "feature_columns": feature_cols,
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "note": "Educational only. Synthetic model.",
    }
    with open(os.path.join(out_dir, "duration_features.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print("Saved duration model artifacts to model_artifacts/")

if __name__ == "__main__":
    main()
