import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt


def plot_frequency_response(
        sr, b, a,
        min_freq=50,
        worN=4196,
        cmap=None,
        min_dB=-10, 
        max_dB=10,
        labels=None,
    ):
    if b.ndim == 1 and a.ndim == 1:
        _b, _a = np.zeros((1, b.shape[0])), np.zeros((1, a.shape[0]))
        _b[0], _a[0] = b, a
        n_plots = 1
    elif b.ndim == 2 and a.ndim == 2 and b.shape[0] == a.shape[0]:
         _b, _a = b.view(), a.view()
         n_plots = b.shape[0]
    else:
        raise ValueError("invalid shapes: {a.shape}, {b.shape}")
    
    if labels is None:
        labels = ["" for _ in range(n_plots)]
    elif type(labels) is not list:
        labels = [labels]
    
    if cmap is None and n_plots < 8:
        cmap = ["r", "b", "g", "c", "m", "y", "k"]
    else:
        cmap = ["b" for _ in range(n_plots)]

    for k in range(n_plots):  
        w, h = signal.freqz(_b[k], _a[k], worN=worN, whole=False, fs=sr)
    
        with np.errstate(divide='ignore'):
            amp_dB = 20 * np.log10(np.abs(h))
        angles = np.angle(h)

        plt.subplot(2,1,1)
        plt.plot(w, amp_dB, color=cmap[k], alpha=0.5, label=labels[k])
        plt.grid(True, which="both", ls="dotted")
        plt.ylabel('Amplitude [dB]')
        plt.xlabel('Frequency [rad/sample]')
        plt.xlim(min_freq, sr)
        plt.ylim(min_dB, max_dB)
        plt.xscale('log')
        if labels[k] != "":
            plt.legend(loc="upper right")

        plt.subplot(2,1,2)
        plt.plot(w, angles, 'g', color=cmap[k], alpha=0.5, label=labels[k])
        plt.grid(True, which="both", ls="dotted")
        plt.ylabel('Angle (radians)')
        plt.xlabel('Frequency [rad/sample]')
        plt.xlim(min_freq, sr)
        plt.ylim(-np.pi, np.pi)
        plt.xscale('log')
        if labels[k] != "":
            plt.legend(loc="upper right")
