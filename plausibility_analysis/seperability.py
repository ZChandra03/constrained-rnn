# hazard_linear_separability.py
"""
Determine which hazard‑rate classes are linearly separable with a purely linear
(LogisticRegression) decoder.  **This version fixes the “Unknown label type:
continuous” error by encoding every hazard value as an *integer class index*.

Quick start
-----------
```powershell
# (inside your venv)
python plausibility_analysis/hazard_linear_separability.py
```
No command‑line arguments are needed.  Adjust `DATA_PATH` below only if your CSV
lives somewhere else.

Outputs
-------
1. **Confusion matrix** of a 21‑way linear classifier.
2. **Heat‑map** of pair‑wise accuracies (one‑vs‑one tests).
3. Printed list of pairs with ≥ 0.90 accuracy (empirically separable).

The script auto‑detects whether your CSV already holds flat numeric columns
(`x0`, `x1`, …) or whether it needs to convert an `evidence` list into a 50‑dim
activation vector (just like your existing decoder).
"""
from __future__ import annotations

import ast
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

##############################################################################
# 0 ─────────────────────────────  Configuration  ────────────────────────────
##############################################################################

DATA_PATH      = Path("plausibility_analysis/variants")  # file **or** directory
FEATURE_PREFIX = "x"      # flat numeric columns start with this
HAZARD_COLUMN  = "hazard"  # falls back to "trueHazard" if absent
WEIGHT_RANGE   = np.linspace(-1, 1, 50)  # for activation reconstruction
TOL            = 1e-3  # float‑matching tolerance when mapping hazards

##############################################################################
# 1 ───────────────────────────  Helper functions  ───────────────────────────
##############################################################################

def compute_activations(ev: list[float], weight_range: np.ndarray = WEIGHT_RANGE):
    """Replicate the toy neuron's forward pass -> 50 activations."""
    h_vec = np.empty(len(weight_range), dtype=np.float32)
    for j, w in enumerate(weight_range):
        h = 0.0
        for x in ev:
            h = np.tanh(w * x + h)
        h_vec[j] = h
    return h_vec


def resolve_data_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        csvs = list(path.glob("*.csv"))
        if len(csvs) == 1:
            return csvs[0]
        raise FileNotFoundError(f"{path} contains {len(csvs)} CSVs – specify one.")
    raise FileNotFoundError(path)

##############################################################################
# 2 ─────────────────────────────  Data loading  ─────────────────────────────
##############################################################################

def load_trials_csv(csv_file: Path, *, feature_prefix: str):
    df = pd.read_csv(csv_file)

    # locate hazard column
    if HAZARD_COLUMN in df.columns:
        hz_col = HAZARD_COLUMN
    elif "trueHazard" in df.columns:
        hz_col = "trueHazard"
    else:
        raise ValueError("No hazard column found (expected 'hazard' or 'trueHazard').")

    # choose feature representation
    feat_cols = [c for c in df.columns if c.startswith(feature_prefix)] if feature_prefix else []

    if feat_cols:
        X = df[feat_cols].to_numpy(float)
    elif "evidence" in df.columns:
        X = np.stack([
            compute_activations(ast.literal_eval(ev_str))
            for ev_str in df["evidence"]
        ])
    else:
        raise ValueError("No flat features and no 'evidence' column found.")

    y_cont = df[hz_col].to_numpy(float)
    return X, y_cont

##############################################################################
# 3 ───────────  Map continuous hazards → integer class indices  ─────────────
##############################################################################

HAZARDS: list[float] = [0.0, 0.05] + [round(0.1 + 0.05 * i, 2) for i in range(19)]
HZ2IDX = {h: i for i, h in enumerate(HAZARDS)}


def match_hazard(v: float, *, tol: float = TOL) -> float:
    for h in HAZARDS:
        if abs(v - h) <= tol:
            return h
    raise ValueError(f"{v} does not match any canonical hazard ±{tol}.")


##############################################################################
# 4 ─────────────────────────────  Main routine  ─────────────────────────────
##############################################################################

def main():
    csv_file = resolve_data_path(DATA_PATH)
    print(f"Loading data from: {csv_file}\n")

    X, y_cont = load_trials_csv(csv_file, feature_prefix=FEATURE_PREFIX)

    # encode hazards as integer classes 0‑20
    y_lab = np.array([HZ2IDX[match_hazard(v)] for v in y_cont], dtype=int)

    # ── 4a  overall multi‑class linear decoder ───────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_lab, test_size=0.25, stratify=y_lab, random_state=0
    )

    clf_multi = LogisticRegression(
        solver="lbfgs", max_iter=100000, n_jobs=-1
    )
    clf_multi.fit(X_tr, y_tr)
    acc_multi = clf_multi.score(X_te, y_te)
    print(f"Overall 21‑way accuracy: {acc_multi:.3f}\n")

    fig1, ax1 = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_te,
        clf_multi.predict(X_te),
        display_labels=[f"{h:.2f}" for h in HAZARDS],
        cmap="Blues",
        xticks_rotation=90,
        ax=ax1,
    )
    ax1.set_title("Confusion matrix (linear decoder, 21 hazards)")
    fig1.tight_layout()

    # ── 4b  pair‑wise separability map ────────────────────────────────────
    n_h = len(HAZARDS)
    acc_mat = np.eye(n_h)

    for i, j in itertools.combinations(range(n_h), 2):
        mask = (y_lab == i) | (y_lab == j)
        if mask.sum() < 10:
            continue
        Xi, yi = X[mask], y_lab[mask]
        yi_bin = (yi == i).astype(int)  # encode pair as 0/1
        Xtr, Xte, ytr, yte = train_test_split(
            Xi, yi_bin, test_size=0.3, stratify=yi_bin, random_state=0
        )
        clf = LogisticRegression(max_iter=10_000, n_jobs=-1)
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        acc_mat[i, j] = acc_mat[j, i] = acc

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im = ax2.imshow(acc_mat, vmin=0.5, vmax=1.0, cmap="viridis")
    ax2.set_xticks(range(n_h))
    ax2.set_xticklabels([f"{h:.2f}" for h in HAZARDS], rotation=90)
    ax2.set_yticks(range(n_h))
    ax2.set_yticklabels([f"{h:.2f}" for h in HAZARDS])
    fig2.colorbar(im, ax=ax2, label="One‑vs‑one test accuracy")
    ax2.set_title("Pair‑wise linear separability heat‑map (≥0.5 = chance)")
    fig2.tight_layout()

    # ── 4c  print easy pairs ──────────────────────────────────────────────
    print("Pairs with accuracy ≥ 0.90 (likely separable):")
    for i, j in itertools.combinations(range(n_h), 2):
        if acc_mat[i, j] >= 0.90:
            print(f"  ({HAZARDS[i]:.2f}, {HAZARDS[j]:.2f})  ->  {acc_mat[i, j]:.3f}")

    plt.show()


if __name__ == "__main__":
    main()
