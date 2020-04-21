from __future__ import division
import numpy as np
from matplotlib import pyplot as plt


def plot_levels(filename):
    with open(filename) as file:
            results = file.readlines()

    scores = np.float64([result.strip().replace('*', '').split() for result in results])

    snrs = np.unique(scores[:, 0])
    n = len(snrs)
    precision = np.empty(n)
    recall = np.empty(n)
    overlap1 = np.empty(n)
    overlap2 = np.empty(n)
    overlap3 = np.empty(n)
    levels = np.empty(n)

    for si, snr in enumerate(snrs):
        idxs = scores[:, 0] == snr
        # ~ print snr,idxs, (scores[idxs,1])
        precision[si] = np.mean(scores[idxs, 1])
        recall[si] = np.mean(scores[idxs, 2])
        overlap1[si] = np.mean(scores[idxs, 5])
        overlap2[si] = np.mean(scores[idxs, 6])
        overlap3[si] = np.mean(scores[idxs, 7])
        levels[si] = np.mean(scores[idxs, 4])

    # plot level overlap
    plt.figure()
    plt.plot(snrs, overlap1, label='overlap 1')
    plt.plot(snrs, overlap2, label='overlap 2')
    plt.plot(snrs, overlap3, label='overlap 3')
    plt.axvline(1, ls=':', color='k', lw=0.5)
    plt.axhline(0, color='k', lw=0.5)
    plt.legend()
    plt.tight_layout()


def plot_complete(filename,title="Symmetric"):
    with open(filename) as file:
            results = file.readlines()

    scores = np.float64([result.split('*')[0].split() for result in results])

    snrs = np.unique(scores[:, 0])
    n = len(snrs)
    precision = np.empty(n)
    recall = np.empty(n)
    precision_std = np.empty(n)
    recall_std = np.empty(n)
    overlap = np.empty(n)
    levels = np.empty(n)

    for si, snr in enumerate(snrs):
        idxs = scores[:, 0] == snr
        # ~ print snr,idxs, (scores[idxs,1])
        precision[si] = np.mean(scores[idxs, 1])
        recall[si] = np.mean(scores[idxs, 2])
        overlap[si] = np.mean(scores[idxs, 3])
        levels[si] = np.mean(scores[idxs, 4])
        precision_std[si] = np.std(scores[idxs, 1])
        recall_std[si] = np.std(scores[idxs, 2])

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.errorbar(snrs, precision, yerr=precision_std, label='Precision')
    ax1.errorbar(snrs, recall, yerr=recall_std, label='Recall')
    ax1.plot(snrs, overlap, label='AMI - Level 1')
    ax1.axvline(1, ls=':', color='r', lw=1)
    ax1.text(1.2,0.9,"Detectability limit",style='italic',rotation=90, color="r")
    ax1.legend()
    ax1.set_xlabel("SNR")
    ax1.set_ylabel("Score")
    ax1.set_ylim(bottom=0)
    ax1.set_title(title)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    ax2.plot(snrs, levels, '*', label='levels', color=color)
    ax2.set_ylabel('# Levels', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()