import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_total_accuracy(model_sliding_accuracy, bayesian_sliding_accuracy):
    plt.plot(model_sliding_accuracy, label="RNN Model", color='blue')
    plt.plot(bayesian_sliding_accuracy, label="Bayesian Observer", color='green')
    plt.axhline(0.5, color='r', linestyle='--', label="Random Baseline (50%)")

    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Accuracy")
    plt.title("Model Accuracy vs Bayesian Observer Accuracy Over Time")
    plt.legend()
    plt.show()

def plot_sample_experiments(X_test, y_test, model):
    num_experiments = 4
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i in range(num_experiments):
        sample_idx = i  
        sample_evidence = X_test[0, sample_idx]  # Shape: (20, 1)
        true_next_drop = y_test[0, sample_idx].item() * 2 -1  # The actual next drop (-1 or 1)
    
        sample_evidence = sample_evidence.unsqueeze(0).unsqueeze(0)
    
        # Get model prediction
        with torch.no_grad():
            output = model(sample_evidence)  # Add batch dim
            predicted_label = torch.argmax(output).item()  # Get class index
    
        # Convert prediction from index (0 or 1) to (-1 or 1)
        predicted_next_drop = -1 if predicted_label == 0 else 1
    
        time_steps = list(range(1, 21))
    
        sample_evidence_flat = sample_evidence.squeeze().numpy().flatten()
    
        # Select subplot axis
        ax = axes[i // 2, i % 2]
    
        # Plot the first 20 data points
        ax.plot(time_steps, sample_evidence_flat, 'bo-', label="Observed Data")
    
        # Plot the 21st time step (True vs Predicted)
        ax.scatter(21, true_next_drop, color='g', marker='o', label="True Next Drop", s=100)
        ax.scatter(21, predicted_next_drop, color='r', marker='x', label="Predicted Next Drop", s=100)
    
        # Add a horizontal line at y=0 for reference
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    
        # Customize plot
        ax.set_xticks(range(1, 22))
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Data Value / Next Drop")
        ax.set_title(f"Test Experiment {i + 1}: Prediction vs True Next Drop")
        ax.legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

def create_bar_plot(labels, rnn_acc, bayes_acc):
    x = np.arange(len(labels))  # [0, 1, 2, 3]
    width = 0.35
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, rnn_acc, width, label='RNN', color='blue')
    rects2 = ax.bar(x + width/2, bayes_acc, width, label='Bayesian', color='green')
    
    # Labels and formatting
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Block Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 100])
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, label='Chance (50%)')
    ax.legend()
    
    # Optionally add value labels on bars
    def add_bar_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_bar_labels(rects1)
    add_bar_labels(rects2)
    
    plt.tight_layout()
    plt.show()
    
def plot_model_behavior_over_time(loss_per_timestep, outputs, evidence):
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    # Plot 1: Objective Function
    axs[0].plot(loss_per_timestep, label='Per-timestep KL Loss', color='orange')
    axs[0].set_ylabel("Loss")
    axs[0].set_title("KL Divergence per Timestep")
    axs[0].legend()

    # Plot 2: Output Probabilities
    axs[1].plot(outputs[:, 0].numpy(), label="P(class = 0)", color='blue')
    axs[1].plot(outputs[:, 1].numpy(), label="P(class = 1)", color='green')
    axs[1].axvline(19, color='gray', linestyle='--', label="Final Time Step")
    axs[1].set_ylabel("Probability")
    axs[1].set_title("Model Prediction Over Time")
    axs[1].legend()

    # Plot 3: Input Evidence
    axs[2].plot(evidence.squeeze().numpy(), marker='o', label="Evidence", color='black')
    axs[2].axhline(0, color='gray', linestyle='--')
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Evidence")
    axs[2].set_title("Input Evidence (Trial Data)")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
    
def plot_network_structure(model):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get trained weights
    W_in = model.fc_input.weight.detach().cpu().numpy().squeeze()     # shape: [hidden_size]
    W_out = model.fc_output.weight.detach().cpu().numpy()             # shape: [2, hidden_size]
    b_out = model.fc_output.bias.detach().cpu().numpy()               # shape: [2]

    hidden_size = W_out.shape[1]

    # Calculate net contribution to each class
    neuron_contribution = {
        "to_-1": W_out[0] * W_in,
        "to_+1": W_out[1] * W_in,
        "diff": (W_out[1] - W_out[0]) * W_in  # discriminatory contribution
    }

    # Sort neurons by absolute contribution difference
    sort_idx = np.argsort(np.abs(neuron_contribution["diff"]))[::-1]

    # --- Plot: Sorted neuron contributions ---
    plt.figure(figsize=(12, 4))
    plt.plot(neuron_contribution["diff"][sort_idx], marker='o', linestyle='-', label="Discriminative Contribution")
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Sorted Hidden Neuron Contributions to Class Separation")
    plt.ylabel("Signed Contribution")
    plt.xlabel("Hidden Units (sorted)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot: Heatmap of top influential neurons ---
    top_k = 50  # Show top 50 most discriminative neurons
    top_idx = sort_idx[:top_k]
    top_contributions = np.stack([W_out[0, top_idx], W_out[1, top_idx]], axis=0)  # shape: [2, top_k]

    plt.figure(figsize=(12, 3))
    sns.heatmap(top_contributions, cmap='coolwarm', center=0, annot=False,
                xticklabels=False, yticklabels=["Class -1", "Class +1"])
    plt.title(f"Top {top_k} Hidden Units: Output Weights (Most Discriminative)")
    plt.xlabel("Hidden Unit Index (sorted by contribution)")
    plt.tight_layout()
    plt.show()

    # --- Plot: Input weights histogram ---
    plt.figure(figsize=(8, 3))
    plt.hist(W_in, bins=50, color='purple', edgecolor='black')
    plt.axvline(0, linestyle='--', color='gray')
    plt.title("Distribution of Input → Hidden Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Number of Hidden Units")
    plt.tight_layout()
    plt.show()

    # --- Plot: Output layer biases ---
    plt.figure(figsize=(4, 2))
    plt.bar(["Class -1", "Class +1"], b_out, color=["blue", "green"])
    plt.axhline(0, linestyle='--', color='gray')
    plt.title("Output Layer Biases")
    plt.ylabel("Bias")
    plt.tight_layout()
    plt.show()
