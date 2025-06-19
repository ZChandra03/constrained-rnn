#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import random

# Seed for reproducibility
SEED = 42


def gen_evidence(n_evidence, hazard_rate, sigma, mu=1, x_lim=5):
    """
    Generate a single sequence of evidence and hidden states under a two-state Markov process.

    Returns:
    - ev (list of float): Noisy observations.
    - state (list of float): Underlying state sequence.
    """
    current_mu = np.random.choice([-mu, mu])
    ev, state = [], []
    for _ in range(n_evidence):
        if sigma > 0:
            lw, hg = -x_lim, x_lim
            a, b = (lw - current_mu) / sigma, (hg - current_mu) / sigma
            sample = truncnorm(a, b, loc=current_mu, scale=sigma).rvs()
        else:
            sample = np.random.normal(current_mu, sigma)
        ev.append(float(sample))
        state.append(current_mu)
        if np.random.rand() < hazard_rate:
            current_mu = -current_mu
    return ev, state


def generate_report_trials(n_trials, hazard_rate, sigma, n_evidence=20):
    """
    Generate a DataFrame of 'report' trials.
    """
    trials = []
    for t in range(1, n_trials + 1):
        ev, state = gen_evidence(n_evidence, hazard_rate, sigma)
        trials.append({'trial': t, 'evidence': ev, 'states': state, 'trueVal': state[-1]})
    return pd.DataFrame(trials)


def compute_mean_activation(evidence_list, weight_range):
    """
    Given a list of evidence sequences and a range of input weights,
    compute the mean final activation across all sequences.
    """
    activations = np.zeros((len(evidence_list), len(weight_range)))
    for i, ev in enumerate(evidence_list):
        for j, w in enumerate(weight_range):
            h = 0.0
            for x in ev:
                h = np.tanh(w * x + h)
            activations[i, j] = h
    return activations.mean(axis=0)


def main():
    # Seed everything for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)

    # Parameters
    n_trials = 100
    sigma = 0.1
    n_evidence = 20
    weight_range = np.linspace(-1, 1, 50)
    hazard_rates = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    plt.figure(figsize=(8, 6))

    for hr in hazard_rates:
        # Generate trials
        df = generate_report_trials(n_trials, hr, sigma, n_evidence)
        # Extract the list of evidence sequences
        evidences = df['evidence'].tolist()
        # Compute mean activation curve
        mean_act = compute_mean_activation(evidences, weight_range)
        # Plot
        plt.plot(weight_range,
                 mean_act,
                 marker='o',
                 label=f'Hazard rate = {hr}')

    plt.xlabel('Input Weight Scaling')
    plt.ylabel('Mean Neuron Activation at Final Timestep')
    plt.title('Mean Activation vs. Input Weight for Different Hazard Rates')
    plt.grid(True)
    plt.legend()
    # Optional: tighten up the x-axis ticks
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
