import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from pyquadfilter import PyQuadFilter
from pyquadfilter import plot_frequency_response


# params
sr = 48000
gain_db = 4.0
pyquad = PyQuadFilter(sr)

filter_type_list = ["lowpass", "highpass", "bandpass", "allpass",
                    "notch", "peaking", "lowshelf", "highshelf"]
q_list  = [0.5, 0.717, 1.0, 1.414, 2.0]
fc_list = [100, 250, 1000, 2500, 5000]

# plot_responses_q
n_patterns = len(q_list) 
b_2dim, a_2dim = np.zeros((n_patterns,3)), np.zeros((n_patterns,3))

for index, filter_type in enumerate(filter_type_list):
    for k, q in enumerate(q_list):
        pyquad.set_params(filter_type, 1000, q, gain_db)
        b_2dim[k, :] = pyquad.b[:]
        a_2dim[k, :] = pyquad.a[:]
    
    plt.clf()
    plot_frequency_response(sr, b_2dim, a_2dim, labels=[f"q={q:.3f}" for q in q_list])
    plt.tight_layout()
    plt.savefig(f"../plot_responses_q/{index}_{filter_type}.png")


# plot_responses_fc
n_patterns = len(fc_list) 
b_2dim, a_2dim = np.zeros((n_patterns,3)), np.zeros((n_patterns,3))

for index, filter_type in enumerate(filter_type_list):
    for k, fc in enumerate(fc_list):
        pyquad.set_params(filter_type, fc, 1.0, gain_db)
        b_2dim[k, :] = pyquad.b[:]
        a_2dim[k, :] = pyquad.a[:]
    
    plt.clf()
    plot_frequency_response(sr, b_2dim, a_2dim, labels=[f"fc={fc:.1f} Hz" for fc in fc_list])
    plt.tight_layout()
    plt.savefig(f"../plot_responses_fc/{index}_{filter_type}.png")



import code
console = code.InteractiveConsole(locals=locals())
console.interact()