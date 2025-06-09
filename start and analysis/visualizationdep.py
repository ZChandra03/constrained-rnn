import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate import evaluate_model_with_tolerance

# Directory to save figures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save_fig(fig, name):
    """
    Save a matplotlib figure to the figures directory and close it.
    """
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_total_accuracy(model_sliding_accuracy, bayesian_sliding_accuracy):
    plt.figure()
    plt.plot(model_sliding_accuracy, label="RNN Model")
    plt.plot(bayesian_sliding_accuracy, label="Bayesian Observer")
    plt.axhline(0.5, linestyle='--', label="Random Baseline (50%)")

    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Accuracy")
    plt.title("Model Accuracy vs Bayesian Observer Accuracy Over Time")
    plt.legend()

    _save_fig(plt.gcf(), "total_accuracy.png")


def plot_sample_experiments(X_test, y_test, model, num_bins):
    num_experiments = 4
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i in range(num_experiments):
        sample_idx = i
        sample_evidence = X_test[0, sample_idx]
        sample_evidence_input = sample_evidence.unsqueeze(0).unsqueeze(0)

        if y_test.ndim == 2:
            true_val = y_test[0, sample_idx].item()
        else:
            true_val = y_test[sample_idx].item()

        if num_bins == 2:
            true_next_drop = -1 if true_val == 0 else 1
        else:
            hazard_bin_center = (true_val + 0.5) / num_bins
            last_obs = sample_evidence[-1].item()
            last_source = -1 if last_obs < 0 else 1
            true_next_drop = -last_source if hazard_bin_center > 0.5 else last_source

        with torch.no_grad():
            output = model(sample_evidence_input)
            predicted_bin = torch.argmax(output).item()
            predicted_hazard = (predicted_bin + 0.5) / num_bins

        last_obs = sample_evidence[-1].item()
        last_source = -1 if last_obs < 0 else 1
        predicted_next_drop = -last_source if predicted_hazard > 0.5 else last_source

        ax = axes[i // 2, i % 2]
        ax.plot(range(1, 21), sample_evidence.squeeze().numpy(), 'bo-', label="Observed Data")
        ax.scatter(21, true_next_drop, color='g', marker='o', label="True Next Drop", s=100)
        ax.scatter(21, predicted_next_drop, color='r', marker='x', label="Predicted Next Drop", s=100)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)

        ax.set_xticks(range(1, 22))
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Data Value / Next Drop")
        ax.set_title(f"Test Experiment {i + 1}: Prediction vs True")
        ax.legend()

    fig.tight_layout()
    _save_fig(fig, "sample_experiments.png")


def plot_accuracy_vs_tolerance(model, X, y_true, max_tolerance=5):
    tolerances = list(range(max_tolerance + 1))
    accuracies = []

    for tol in tolerances:
        _, acc = evaluate_model_with_tolerance(model, X, y_true, tolerance=tol)
        accuracies.append(acc * 100)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(tolerances, accuracies, marker='o')
    plt.xlabel("Tolerance (± bins)")
    plt.ylabel("Accuracy (%)")
    plt.title("RNN Accuracy vs Bin Tolerance")
    plt.grid(True)
    plt.tight_layout()

    _save_fig(fig, "accuracy_vs_tolerance.png")


def create_bar_plot(labels, rnn_acc, bayes_acc):
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, rnn_acc, width, label='RNN')
    rects2 = ax.bar(x + width/2, bayes_acc, width, label='Bayesian')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy of Next-Drop Prediction by Block Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 100])
    ax.axhline(50, linestyle='--', linewidth=1, label='Chance (50%)')
    ax.legend()

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    fig.tight_layout()
    _save_fig(fig, "bar_accuracy.png")


def plot_model_behavior_over_time(loss_per_timestep, outputs, evidence):
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(loss_per_timestep, label='Per-timestep KL Loss')
    axs[0].set_ylabel("Loss")
    axs[0].set_title("KL Divergence per Timestep")
    axs[0].legend()

    axs[1].plot(outputs[:, 0].numpy(), label="P(class = 0)")
    axs[1].plot(outputs[:, 1].numpy(), label="P(class = 1)")
    axs[1].axvline(19, linestyle='--', label="Final Time Step")
    axs[1].set_ylabel("Probability")
    axs[1].set_title("Model Prediction Over Time")
    axs[1].legend()

    axs[2].plot(evidence.squeeze().numpy(), marker='o', label="Evidence")
    axs[2].axhline(0, linestyle='--')
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Evidence")
    axs[2].set_title("Input Evidence (Trial Data)")
    axs[2].legend()

    fig.tight_layout()
    _save_fig(fig, "model_behavior.png")


def plot_accuracy_vs_hazard(bin_centers, rnn_acc, bayes_acc, bin_counts,
                            title="Accuracy vs Hazard Rate", line=50):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(bin_centers, rnn_acc, marker='o', label="RNN")
    ax1.plot(bin_centers, bayes_acc, marker='o', label="Bayesian")
    ax1.axhline(line, linestyle='--', label=f"Chance ({100 // len(bin_centers)}%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    bar_width = (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 0.05
    ax2.bar(bin_centers, bin_counts, width=bar_width, edgecolor='black')
    ax2.set_ylabel("Trial Count")
    ax2.set_xlabel("True Hazard Rate")
    ax2.set_ylim(bottom=0)
    ax2.grid(True)

    fig.tight_layout()
    _save_fig(fig, "accuracy_vs_hazard.png")


def print_next_drop_stats(name, accuracy, predicted_next):
    num_stay = (predicted_next == -1).sum().item()
    num_switch = (predicted_next == 1).sum().item()
    total = len(predicted_next)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print(f"  Decisions to STAY  (-1): {num_stay} ({num_stay / total * 100:.2f}%)")
    print(f"  Decisions to SWITCH (+1): {num_switch} ({num_switch / total * 100:.2f}%)")


def plot_network_structure(model, num_classes):
    W_in = model.fc_input.weight.detach().cpu().numpy().squeeze()
    W_out = model.fc_output.weight.detach().cpu().numpy()
    b_out = model.fc_output.bias.detach().cpu().numpy()

    hidden_size = W_out.shape[1]
    neuron_contribution = {
        "to_-1": W_out[0] * W_in,
        "to_+1": W_out[1] * W_in,
        "diff": (W_out[1] - W_out[0]) * W_in
    }
    sort_idx = np.argsort(np.abs(neuron_contribution["diff"]))[::-1]

    # Sorted contribution plot
    fig = plt.figure(figsize=(12, 4))
    plt.plot(neuron_contribution["diff"][sort_idx], marker='o')
    plt.axhline(0, linestyle='--')
    plt.title("Sorted Hidden Neuron Contributions to Class Separation")
    plt.ylabel("Signed Contribution")
    plt.xlabel("Hidden Units (sorted)")
    fig.tight_layout()
    _save_fig(fig, "network_contributions.png")

    # Heatmap of top influential neurons
    top_k = 50
    top_idx = sort_idx[:top_k]
    top_contributions = np.stack([W_out[0, top_idx], W_out[1, top_idx]], axis=0)

    fig = plt.figure(figsize=(12, 3))
    sns.heatmap(top_contributions, center=0, annot=False, xticklabels=False,
                yticklabels=["Class -1", "Class +1"])
    plt.title(f"Top {top_k} Hidden Units: Output Weights")
    fig.tight_layout()
    _save_fig(fig, "network_heatmap.png")

    # Input weight histogram
    fig = plt.figure(figsize=(8, 3))
    plt.hist(W_in, bins=50, edgecolor='black')
    plt.axvline(0, linestyle='--')
    plt.title("Distribution of Input → Hidden Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Number of Hidden Units")
    fig.tight_layout()
    _save_fig(fig, "input_weight_histogram.png")
