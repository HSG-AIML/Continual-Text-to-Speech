import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


def plot_attention(attn):
    """Plot attention."""
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation='nearest', aspect='auto')
    return fig


def plot_spectrogram(M, length=None):
    """Plot spectrogram."""
    M = np.flip(M, axis=0)
    if length:
        M = M[:, :length]
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    return fig


def plot_spectrogram_double(M1, M2, length_spec=None):
    """Plot two spectrograms together."""
    M1 = np.flip(M1, axis=0)
    M2 = np.flip(M2, axis=0)
    if length_spec:
        M1 = M1[:, :length_spec]
        M2 = M2[:, :length_spec]

    fig, ax = plt.subplots(2)
    fig.set_figheight(10)
    fig.set_figwidth(12)

    ax[0].imshow(M1, interpolation='nearest', aspect='auto')
    ax[1].imshow(M2, interpolation='nearest', aspect='auto')

    return fig
