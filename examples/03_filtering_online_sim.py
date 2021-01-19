import os
import sys
sys.path.append("../")
import soundfile as sf
from pyquadfilter import PyQuadFilter


filter_type_list = ["lowpass", "highpass", "bandpass", "allpass",
                    "notch", "peaking", "lowshelf", "highshelf"]

# Params
gain_db = 4.0
q       = 1.0
fc      = 1000
filter_type = filter_type_list[0]

# Filtering
filepath = "audio/miku_doremi_bpm120.wav"
x, sr = sf.read(filepath)
x = x.T 
print(x.shape) # =>(n_ch, n_samples)

pyquad = PyQuadFilter(sr)
pyquad.set_params(filter_type, fc, q, gain_db)

# Sim. online (frame-by-frame proc.)
length = 256
y = x.copy() * 0.0
n_ch, n_samples = x.shape
pyquad.prepare_filter_online(n_ch=n_ch, length=length)

for k in range(n_samples // 256):
    sl = slice(k*length, (k+1)*length)
    y[:, sl] = pyquad.filter(x[:, sl], online=True)

# Save y
basename, ext = os.path.splitext(filepath) 
sf.write(f"{basename}_filtered{ext}", y.T, sr)
