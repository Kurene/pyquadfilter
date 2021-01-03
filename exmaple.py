import numpy as np
import matplotlib.pyplot as plt
from pyquadfilter import PyQuadFilter
from pyquadfilter import plot_frequency_response


sr = 48000
pyquad = PyQuadFilter(sr, "lowpass")
plot_frequency_response(sr, pyquad.b, pyquad.a)


gain_db = 4.0
filter_type_list = ["lowpass", "highpass", "bandpass", "allpass",
                    "notch", "peaking", "lowshelf", "highshelf"]
q_list = [0.5, 0.717, 1.0, 1.414, 2.0]
fc_list = [100, 250, 1000, 2500, 5000]

n_patterns = len(q_list) 
b_2dim, a_2dim = np.zeros((n_patterns,3)), np.zeros((n_patterns,3))

for filter_type in filter_type_list:
    for index, q in enumerate(q_list):
        pyquad.set_params(filter_type, 1000, q, gain_db)
        b_2dim[index, :] = pyquad.b[:]
        a_2dim[index, :] = pyquad.a[:]
    
    plt.clf()
    plot_frequency_response(sr, b_2dim, a_2dim, label=[f"q={q:.1f}" for q in q_list])
    plt.tight_layout()
    plt.savefig(f"{filter_type}.png")

import code
console = code.InteractiveConsole(locals=locals())
console.interact()