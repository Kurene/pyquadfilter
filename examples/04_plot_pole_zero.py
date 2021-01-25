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
gain_db = 6.0
q       = 1.5
fc      = 1000
filter_type = filter_type_list[3]

pyquad = PyQuadFilter(sr)
pyquad.set_params(filter_type, fc, q, gain_db)

pyquad.plot_pole_zero()
plt.show()