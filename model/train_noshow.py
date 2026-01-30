# model/train_noshow.py
"""
Educational Use Only (Non-Clinical)
Trains a simple MLP classifier to predict outpatient no-show probability.
Exports:
- model_artifacts/noshow_model.pt
- model_artifacts/noshow_features.json  (feature column order for UI)
"""

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def make_features(df: pd.DataFrame, y_col: str, feature_columns: list[str] | None = None):
    cat_cols = ["resource_modality", "exam_code", "priority", "scanner_id"]
    num_cols = ["requires_contrast", "requires_sedation", "isolation", "hour", "dow", "lead_time_days"]

    for c in cat_cols + num_cols + [y_col]:
        if c not in df.columns:
            df[c] = 0

    X_cat = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols)
    X_num = df[num_cols].astype(float)
    X = pd.concat([X_num, X_cat], axis=1)

    if feature_columns is None:
        feature_columns = X.columns.tolist()
    else:
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_columns]

    y = df[y_col].astype(int).to_numpy()
    X = X.to_numpy(dtype=np.float32)
    y = y.astype(np.float32)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-6] = 1.0
    Xn = (X - X_mean) / X_std

    return Xn, y, feature_columns, X_mean, X_std


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


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(repo_root, "data", "derived", "train_noshow.csv")
    out_dir = os.path.join(repo_root, "model_artifacts")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path).copy()
    # Only outpatient table should be here, but keep safe:
    if "no_show" not in df.columns:
        raise ValueError("train_noshow.csv missing no_show column")

    rng = np.random.default_rng(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(0.85 * len(df))
    tr_idx, va_idx = idx[:split], idx[split:]

    Xtr, ytr, feature_cols, X_mean, X_std = make_features(df.iloc[tr_idx], "no_show")
    Xva, yva, _, _, _ = make_features(df.iloc[va_idx], "no_show", feature_cols)

    device = "cpu"
    model = MLPClassifier(in_dim=Xtr.shape[1]).to(device)

    # class imbalance weight
    pos = max(1.0, float(ytr.sum()))
    neg = max(1.0, float((1 - ytr).sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=512, shuffle=True
    )

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    model.train()
    for epoch in range(8):
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            vlogits = model(torch.from_numpy(Xva).to(device)).cpu().numpy()
        vprob = sigmoid(vlogits)
        # simple AUC-free metric: logloss + accuracy at 0.5
        eps = 1e-7
        logloss = float(-np.mean(yva * np.log(vprob + eps) + (1 - yva) * np.log(1 - vprob + eps)))
        acc = float(np.mean((vprob >= 0.5) == (yva >= 0.5)))
        model.train()
        print(f"epoch {epoch+1}/8  train_loss={np.mean(losses):.4f}  val_logloss={logloss:.4f}  val_acc={acc:.3f}")

    torch.save(model.state_dict(), os.path.join(out_dir, "noshow_model.pt"))

    meta = {
        "feature_columns": feature_cols,
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "note": "Educational only. Synthetic model.",
    }
    with open(os.path.join(out_dir, "noshow_features.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print("Saved no-show model artifacts to model_artifacts/")

if __name__ == "__main__":
    main()
