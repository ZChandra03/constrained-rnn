# ==============================================================
#  Project: Hazard‑rate inference task – FULL UPDATED CODE BASE
#  Third revision – convert_subset_to_tensor now **always returns 4‑D**
#  tensors so rnn.forward never receives an empty or 3‑D input.
# ==============================================================

# -------------------------------------------------------------------
# data_manager.py
# -------------------------------------------------------------------

import torch
import pandas as pd
import numpy as np
import random


def load_data(params, shuffle=True):
    """(unchanged – still returns sigmas_*, hazards_*, etc.)"""
    all_data = []
    easy_data, medium_data, hard_data, pretest_data = [], [], [], []

    hazards_train, hazards_test = [], []
    hazards_easy, hazards_medium, hazards_hard, hazards_pretest = [], [], [], []

    true_next_train, true_next_test = [], []

    # σ per‑trial
    sigmas_train, sigmas_test = [], []
    sigmas_easy, sigmas_medium, sigmas_hard, sigmas_pretest = [], [], [], []

    for k in range(params["variants"]):
        train_df = pd.read_csv(f"variants/trainConfig_var{k}.csv")
        test_df  = pd.read_csv(f"variants/testConfig_var{k}.csv")

        train_df = train_df[train_df.iloc[:, 4] == "predict"]
        test_df  = test_df [test_df .iloc[:, 4] == "predict"]

        # ---------------- TRAIN ---------------- #
        evidence_train = train_df["evidence"].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))
        true_val_train = train_df["trueVal"].values
        hazard_train   = train_df["trueHazard"].values
        sigma_train    = train_df["sigma"].values

        hazards_train.extend(hazard_train.tolist())
        sigmas_train .extend(sigma_train .tolist())
        true_next_train.extend((true_val_train == 1).astype(int).tolist())

        all_data.append(
            (torch.tensor(np.stack(evidence_train), dtype=torch.float32).unsqueeze(-1),
             torch.tensor((true_val_train == 1).astype(int), dtype=torch.long))
        )

        # ---------------- TEST ----------------- #
        evidence_test = test_df["evidence"].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))
        true_val_test = test_df["trueVal"].values
        hazard_test   = test_df["trueHazard"].values
        sigma_test    = test_df["sigma"].values

        hazards_test.extend(hazard_test.tolist())
        sigmas_test .extend(sigma_test .tolist())
        true_next_test.extend((true_val_test == 1).astype(int).tolist())

        all_data.append(
            (torch.tensor(np.stack(evidence_test), dtype=torch.float32).unsqueeze(-1),
             torch.tensor((true_val_test == 1).astype(int), dtype=torch.long))
        )

        # ---- split test by difficulty ---- #
        difficulty = test_df.iloc[:, 2].values
        for ev, val, diff, hz, sg in zip(
                evidence_test.tolist(), true_val_test, difficulty, hazard_test, sigma_test):
            sample = (
                torch.tensor(ev, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(int(val == 1), dtype=torch.long),
            )
            if diff == "easy":
                easy_data.append(sample);   hazards_easy.append(hz); sigmas_easy.append(sg)
            elif diff == "medium":
                medium_data.append(sample); hazards_medium.append(hz); sigmas_medium.append(sg)
            elif diff == "hard":
                hard_data.append(sample);   hazards_hard.append(hz); sigmas_hard.append(sg)
            elif diff == "preTest":
                pretest_data.append(sample); hazards_pretest.append(hz); sigmas_pretest.append(sg)

    # ------------- shuffle (safely) ------------- #
    if shuffle:
        for dataset, hz_list, sg_list in [
            (easy_data, hazards_easy, sigmas_easy),
            (medium_data, hazards_medium, sigmas_medium),
            (hard_data, hazards_hard, sigmas_hard),
            (pretest_data, hazards_pretest, sigmas_pretest),
        ]:
            paired = list(zip(dataset, hz_list, sg_list))
            if paired:  # skip empty difficulty
                random.shuffle(paired)
                dataset[:], hz_list[:], sg_list[:] = zip(*paired)

    return (
        all_data,
        easy_data, medium_data, hard_data, pretest_data,
        np.array(hazards_train),  np.array(hazards_test),
        np.array(hazards_easy),   np.array(hazards_medium),
        np.array(hazards_hard),   np.array(hazards_pretest),
        np.array(sigmas_train),   np.array(sigmas_test),
        np.array(sigmas_easy),    np.array(sigmas_medium),
        np.array(sigmas_hard),    np.array(sigmas_pretest),
        torch.tensor(true_next_train, dtype=torch.long),
        torch.tensor(true_next_test,  dtype=torch.long),
    )


# -------------------------------------------------------------------
# NEW: convert_subset_to_tensor – always returns 4‑D X
# -------------------------------------------------------------------

def convert_subset_to_tensor(subset, variants, seq_len: int = 20):
    """Return tensors ready for the RNN.

    Output shapes
    -------------
    X_tensor : [num_experiments, batch_size, seq_len, 1]
    y_tensor : [num_experiments * batch_size]
    """

    if len(subset) == 0:
        X_tensor = torch.empty((1, 0, seq_len, 1))  # 1 experiment, 0 batch
        y_tensor = torch.empty((0,), dtype=torch.long)
        return X_tensor, y_tensor

    # -- stack --
    X_list, y_list = zip(*[(ev, lbl) for ev, lbl in subset])
    X_tensor = torch.stack(X_list)  # [N, seq_len, 1]
    y_tensor = torch.stack(y_list)  # [N]

    # ensure 3‑D evidence (seq_len last)
    if X_tensor.dim() == 2:
        X_tensor = X_tensor.unsqueeze(-1)

    # reshape by variants when possible
    if len(subset) % variants == 0 and variants > 1:
        X_tensor = X_tensor.view(variants, -1, seq_len, 1)  # [V, B, 20, 1]
        y_tensor = y_tensor.view(variants, -1)              # [V, B]
    # Only add an experiment dimension if it is still missing
    elif X_tensor.dim() == 3:                               # [N, 20, 1]
        X_tensor = X_tensor.unsqueeze(0)                    # [1, N, 20, 1]
        y_tensor = y_tensor.unsqueeze(0)                    # [1, N]

    return X_tensor, y_tensor
