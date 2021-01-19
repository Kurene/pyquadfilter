import os
import sys
sys.path.append("../")

import soundfile as sf
import matplotlib.pyplot as plt
from pyquadfilter import PyQuadFilter


filter_type_list = ["lowpass", "highpass", "bandpass", "allpass",
                    "notch", "peaking", "lowshelf", "highshelf"]

# params
sr      = 48000
gain_db = 4.0
q       = 1.0
fc      = 1000
filter_type = filter_type_list[0]

pyquad = PyQuadFilter(sr)
pyquad.set_params(filter_type, fc, q, gain_db)

plt.clf()
pyquad.plot_frequency_response()
plt.tight_layout()
plt.show()
