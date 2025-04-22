import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from NormativeModel import BayesianObserver, evaluate_bayesian_accuracy
from rnn import HazardRNN, evaluate_model
from data_manager import load_data, convert_subset_to_tensor
from visualization import plot_total_accuracy, create_bar_plot, plot_model_behavior_over_time, plot_network_structure, plot_sample_experiments


# Training setup
params = {'variants': 10}
num_classes = 2
data, easy, medium, hard, pretest = load_data(params)
print('Data loaded from file.')

X_train, y_train = convert_subset_to_tensor(data[::2], params['variants'])
X_test, y_test = convert_subset_to_tensor(data[1::2], params['variants'])
X_easy, y_easy = convert_subset_to_tensor(easy, params['variants'])
X_med, y_med = convert_subset_to_tensor(medium, params['variants'])
X_hard, y_hard = convert_subset_to_tensor(hard, params['variants'])
X_pretest, y_pretest = convert_subset_to_tensor(pretest, params['variants'])

print('Data processed for classification!')

# Define model, loss, and optimizer
model = HazardRNN(output_size=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Toggle silent or instant output
training_scheme = "silent"

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    if training_scheme == "instant":
        outputs = model(X_train)  # [batch*experiments, 2]
        loss = criterion(outputs, y_train)

    elif training_scheme == "silent":
        outputs = model(X_train, return_all=True)  # [batch*experiments, seq_len, 2]
        batch_size = outputs.size(0)
        seq_len = outputs.size(1)

        # Create soft targets: [batch*experiments, seq_len, 2]
        target_probs = torch.full_like(outputs, 0)  # default = [0.5, 0.5]
        target_probs[:, -1, :] = F.one_hot(y_train, num_classes=num_classes).float().view(-1,num_classes) # real target at final step

        # Compute loss only on final step and neutral targets otherwise
        loss = F.kl_div(outputs.log(), target_probs, reduction='batchmean')  # KL divergence for soft targets

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")


# Compare predictions with true labels
correct, accuracy = evaluate_model(model, X_test, y_test) 
easy_correct, easy_accuracy = evaluate_model(model, X_easy, y_easy) 
med_correct, med_accuracy  = evaluate_model(model, X_med, y_med) 
hard_correct, hard_accuracy = evaluate_model(model, X_hard, y_hard) 
pretest_correct, pretest_accuracy = evaluate_model(model, X_pretest, y_pretest) 

print(f"Total Accuracy: {accuracy * 100:.2f}%")
print(f"Easy Accuracy: {easy_accuracy * 100:.2f}%")
print(f"Medium Accuracy: {med_accuracy * 100:.2f}%")
print(f"Hard Accuracy: {hard_accuracy * 100:.2f}%")
print(f"Pretest Accuracy: {pretest_accuracy * 100:.2f}%")


# Compute sliding average accuracy for the model (window size = 100)
window_size = 50
# Pad with zeros on both sides for centered window
model_sliding_accuracy = F.avg_pool1d(correct.view(1, 1, -1), kernel_size=window_size, stride=1, padding=window_size//2)
model_sliding_accuracy = model_sliding_accuracy.squeeze().numpy()



print("Running Bayesian Model...")
bayesian_cumulative_accuracy = []
bayesian_correct = []

# Run Bayesian Observer on the test set (assuming y_test is the evidence for simplicity)
hs = np.arange(0, 1, 0.05)
y_test_converted = 2 * y_test - 1
for i in range(X_test.size(0) * X_test.size(1)):
    experiment_idx = i // X_test.size(1)
    sample_idx = i % X_test.size(1)

    ev = X_test[experiment_idx, sample_idx, :, 0].numpy().tolist()
    #print(ev)

    _, _, rep, pred = BayesianObserver(ev, mu1=-1, mu2=1, sigma=0.1, hs=hs)
    
    # Compare Bayesian predictions to true labels
    is_correct = (pred == y_test_converted[experiment_idx, sample_idx].item())
    bayesian_correct.append(is_correct)

# Run per-difficulty evaluation
easy_bayes_acc = evaluate_bayesian_accuracy(easy, hs)
med_bayes_acc = evaluate_bayesian_accuracy(medium, hs)
hard_bayes_acc = evaluate_bayesian_accuracy(hard, hs)
pretest_bayes_acc = evaluate_bayesian_accuracy(pretest, hs)
    
# Convert to numpy array
bayesian_correct = np.array(bayesian_correct, dtype=np.float32)

bayesian_sliding_accuracy = np.convolve(
    bayesian_correct, np.ones(window_size)/window_size, mode='same'
)

print("Complete! Generating Reuslts.")


plot_total_accuracy(model_sliding_accuracy, bayesian_sliding_accuracy)

plot_sample_experiments(X_test, y_test, model)

# Choose a sample index to visualize
sample_idx = 1

# Get one trial's input (shape: [20, 1])
evidence = X_test[0, sample_idx]

evidence = evidence.unsqueeze(0).unsqueeze(0)  # Shape: [1, 20, 1, 1]
target_class = y_train[0, sample_idx].item()

# Forward pass with return_all=True
model.eval()
with torch.no_grad():
    outputs = model(evidence, return_all=True)  # [1, seq_len, 2]
    outputs = outputs.squeeze(0)  # [seq_len, 2]

# Create target probs: neutral until last step
target_probs = torch.full_like(outputs, 0)
target_probs[-1] = F.one_hot(torch.tensor(target_class), num_classes=num_classes).float()

# Compute per-timestep loss (KL divergence)
loss_per_timestep = F.kl_div(outputs.log(), target_probs, reduction='none').sum(dim=1).numpy()

plot_model_behavior_over_time(loss_per_timestep, outputs, evidence)

plot_network_structure(model)

# Create bar plot
# Difficulty levels
labels = ['easy', 'medium', 'hard', 'preTest']

# Accuracy values (as percentages)
rnn_acc = [
    easy_accuracy * 100,
    med_accuracy * 100,
    hard_accuracy * 100,
    pretest_accuracy * 100,
]
bayes_acc = [
    easy_bayes_acc * 100,
    med_bayes_acc * 100,
    hard_bayes_acc * 100,
    pretest_bayes_acc * 100,
]

create_bar_plot(labels, rnn_acc, bayes_acc)
