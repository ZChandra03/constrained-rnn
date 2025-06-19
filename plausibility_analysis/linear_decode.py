#!/usr/bin/env python3
"""
Linear decoder for hazard rate with manual‐bin support.

This script will:
  1) load variants/testConfig_var0.csv
  2) for each bin count in BIN_LIST, discretise the hazards either by:
       • your manually‐specified edges, or
       • equal‐width centres if none provided
     train a multinomial logistic decoder, and print accuracy + per‐class report.

Usage:
    python linear_decode.py
"""

import os
import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────
VARIANTS_FILE    = "variants/testConfig_var0.csv"
BIN_LIST         = [2]
MANUAL_BIN_EDGES = {
    2: [0.0, 0.5, 1.0],
    # for 3 bins: edges at 0.0, 0.3, 0.7, 1.0
    3: [0.0, 0.31, 0.69, 1.0],
    # for 7 bins: edges at 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0
    7: [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0],
    # 21 not specified here → will use equal‐width centres at 0.00, 0.05, …, 1.00
}
TEST_SIZE        = 0.2
RANDOM_STATE     = 0

# ────────────────────────────────────────────────────────────────────────
# 1. Neuron model: compute a 50‐unit tanh activation for one trial
# ────────────────────────────────────────────────────────────────────────
WEIGHT_RANGE = np.linspace(-1, 1, 50)

def compute_activations(ev, weight_range=WEIGHT_RANGE):
    acts = np.empty(len(weight_range), dtype=np.float32)
    for j, w in enumerate(weight_range):
        h = 0.0
        for x in ev:
            h = np.tanh(w * x + h)
        acts[j] = h
    return acts

# ────────────────────────────────────────────────────────────────────────
# 2. Load just the single CSV into X (activations) and y_cont (true hazard)
# ────────────────────────────────────────────────────────────────────────
def load_trials(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path       = os.path.join(script_dir, file_name)
    df         = pd.read_csv(path)

    X, y_cont = [], []
    for ev_str, hz in zip(df["evidence"], df["trueHazard"]):
        ev = ast.literal_eval(ev_str)
        X.append(compute_activations(ev))
        y_cont.append(float(hz))

    return np.stack(X), np.asarray(y_cont)

# ────────────────────────────────────────────────────────────────────────
# 3a. Equal‐width (nearest‐centre) discretisation
# ────────────────────────────────────────────────────────────────────────
def discretise_equal_width(y_cont, n_bins):
    centres = np.linspace(0, 1, n_bins)
    y_class = np.abs(y_cont[:, None] - centres[None, :]).argmin(axis=1)
    return y_class, centres

# ────────────────────────────────────────────────────────────────────────
# 3b. Manual‐edges discretisation via np.digitize
# ────────────────────────────────────────────────────────────────────────
def discretise_manual(y_cont, edges):
    # edges: list of length n_bins+1, sorted, from 0 to 1
    edges = np.array(edges)
    # drop first/last for digitize
    bins  = edges[1:-1]
    # np.digitize returns 0..len(bins), exactly your class labels
    y_class = np.digitize(y_cont, bins)
    centres = (edges[:-1] + edges[1:]) / 2
    return y_class, centres

# ────────────────────────────────────────────────────────────────────────
# 4. Main: loop over BIN_LIST, train & report
# ────────────────────────────────────────────────────────────────────────
def main():
    X, y_cont = load_trials(VARIANTS_FILE)

    for n_bins in BIN_LIST:
        print(f"\n=== Results with {n_bins} bins ===")
        if n_bins in MANUAL_BIN_EDGES:
            y_class, centres = discretise_manual(y_cont, MANUAL_BIN_EDGES[n_bins])
        else:
            y_class, centres = discretise_equal_width(y_cont, n_bins)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_class,
            test_size    = TEST_SIZE,
            random_state = RANDOM_STATE,
            stratify     = y_class
        )

        # start with unit weight for every class…
        center_idx = 1
        cw = {i: 1.0 for i in range(n_bins)}
        cw[center_idx] = 0.0

        clf = LogisticRegression(
            max_iter     = 3000,
            solver       = "lbfgs",
            class_weight = 'balanced',
        )
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        print(f"Overall accuracy ({n_bins} bins): {acc:.3f}\n")
        print(classification_report(
            y_te,
            y_pred,
            target_names = [f"{c:.2f}" for c in centres],
            zero_division= 0
        ))

if __name__ == "__main__":
    main()
