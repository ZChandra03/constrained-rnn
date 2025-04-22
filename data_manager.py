import torch
import pandas as pd
import numpy as np

def load_data(params):
    all_data = []
    easy_data = []
    medium_data = []
    hard_data = []
    pretest_data = []
    for k in range(params['variants']):
        train_data = pd.read_csv(f"variants/trainConfig_var{k}.csv")
        test_data = pd.read_csv(f"variants/testConfig_var{k}.csv")
        
        # Only use the prediction mode for now (ignore report)
        train_data = train_data[train_data.iloc[:, 4] == 'predict']
        test_data = test_data[test_data.iloc[:, 4] == 'predict']

        # Extract evidence and true values directly
        evidence_train = train_data['evidence'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        true_val_train = train_data['trueVal'].values  # Directly use 'trueVal' column

        evidence_test = test_data['evidence'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        true_val_test = test_data['trueVal'].values  # Directly use 'trueVal' column


        all_data.append((torch.tensor(np.stack(evidence_train), dtype=torch.float32).unsqueeze(-1),
                         torch.tensor((true_val_train == 1).astype(int), dtype=torch.long)))  # Convert -1/1 to 0/1

        all_data.append((torch.tensor(np.stack(evidence_test), dtype=torch.float32).unsqueeze(-1),
                         torch.tensor((true_val_test == 1).astype(int), dtype=torch.long)))  # Convert -1/1 to 0/1
        
        # --- TEST DATA, by difficulty ---
        true_val_test = test_data['trueVal'].values
        difficulty = test_data.iloc[:, 2].values  # Column 3: difficulty

        for ev, val, diff in zip(evidence_test.tolist(), true_val_test, difficulty):
            sample = (torch.tensor(ev, dtype=torch.float32).unsqueeze(-1),
                      torch.tensor(int(val == 1), dtype=torch.long))

            if diff == 'easy':
                easy_data.append(sample)
            elif diff == 'medium':
                medium_data.append(sample)
            elif diff == 'hard':
                hard_data.append(sample)
            elif diff == 'preTest':
                pretest_data.append(sample)

    return all_data, easy_data, medium_data, hard_data, pretest_data

def convert_subset_to_tensor(subset, variants):
    """
    Converts a list of (evidence_tensor, label_tensor) pairs into X and y tensors
    shaped like X_test and y_test in your workflow.
    """
    X_list, y_list = zip(*subset)
    X_tensor = torch.stack(X_list).unsqueeze(-1)  # Shape: [N, 20, 1, 1]
    y_tensor = torch.stack(y_list)                # Shape: [N]
    
    X_tensor = X_tensor.view(variants, -1, 20, 1)
    
    return X_tensor, y_tensor