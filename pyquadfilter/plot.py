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
          


def plot_pole_zero(b, a):
    max_v = 1.0
    def plot_comp(c, label, color, marker, _max_v):
        x, y = np.real(c), np.imag(c)
        plt.plot(x, y, marker=marker, color=color, label=label)
        _max_v = np.abs(x) if np.abs(x) > _max_v else _max_v
        _max_v = np.abs(y) if np.abs(y) > _max_v else _max_v
        return _max_v
        
    plt.figure(figsize=(6, 6))
    theta = np.linspace(0, 2*np.pi, 360)
    plt.plot(np.cos(theta), np.sin(theta), c="black", linestyle="dotted")
    plt.grid()
    
    zeros, poles = np.roots(b), np.roots(a)
    max_v = plot_comp(zeros[0], "zero 0", "b", "x", max_v)
    max_v = plot_comp(zeros[1], "zero 1", "b", "x", max_v)
    max_v = plot_comp(poles[0], "pole 0", "r", "o", max_v)
    max_v = plot_comp(poles[1], "pole 1", "r", "o", max_v)
    max_v *= 1.1
    plt.xlim(-max_v, max_v)
    plt.ylim(-max_v, max_v)
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
    
if __name__ == "__main__":
    
    
    b = np.array([1.0, 2.0, 1.0])
    a = np.array([5.0, -1.0, 1.0])
    
    plot_pole_zero(b, a)
    plt.show()
    
    
    
    