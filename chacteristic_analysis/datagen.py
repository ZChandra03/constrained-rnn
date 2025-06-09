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


def plot_first_trial(evidence):
    """
    Scatter-plot the evidence sequence of the first trial,
    with x-axis ticks locked to increments of 1 and margins added.
    """
    time = np.arange(len(evidence))
    fig, ax = plt.subplots()
    ax.scatter(time, evidence)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Evidence Value')
    ax.set_title('First Trial Evidence Sequence')
    ax.grid(True)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.margins(x=0.05)
    plt.show()


def plot_activation_vs_weight(evidence, weight_range=None):
    """
    Plot final hidden activation of an imaginary neuron (no bias) versus input weight scaling.

    - Uses the first trial's evidence sequence.
    - Sweeps weight_range (defaults to 50 points from -1 to 1).
    - Neuron update: h_t = tanh(w * x_t + h_{t-1}).
    """
    if weight_range is None:
        weight_range = np.linspace(-1, 1, 50)
    activations = []
    for w in weight_range:
        h = 0.0
        for x in evidence:
            h = np.tanh(w * x + h)
        activations.append(h)
    fig, ax = plt.subplots()
    ax.plot(weight_range, activations, marker='o')
    ax.set_xlabel('Input Weight Scaling')
    ax.set_ylabel('Neuron Activation at Final Timestep')
    ax.set_title('Activation vs Input Weight')
    ax.grid(True)
    plt.show()


def main():
    # Seed everything for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)

    # Hardcoded parameters
    n_trials = 50
    hazard_rate = 0.8
    sigma = 0.1
    n_evidence = 20
    trial = 1

    # Generate trials and plot
    df = generate_report_trials(n_trials, hazard_rate, sigma, n_evidence)
    evidence = df.iloc[trial]['evidence']
    plot_first_trial(evidence)
    plot_activation_vs_weight(evidence)


if __name__ == '__main__':
    main()
