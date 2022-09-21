# PyQuadFilter

Implementation of digital bi-quad filter in Python

reference:
https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html

使い方 (日本語)
https://www.wizard-notes.com/entry/python/pyquadfilter-prototype

# Quick Start

## Installation
```
pip install git+https://github.com/kurene/pyquadfilter
```

## Compute bi-qiad filter coefs
```python
from pyquadfilter import PyQuadFilter

pyquad = PyQuadFilter(sr)
pyquad.set_params(filter_type, fc, q, gain_db)
print(pyquad.b)
print(pyquad.a) 
```

or

```python
from pyquadfilter import PyQuadFilter

pyquad = PyQuadFilter(sr, filter_type=filter_type, fc=fc, q=q, gain_db=gain_db)
```

## Filter the signal
### Offline
```python
pyquad = PyQuadFilter(sr)
pyquad.set_params(filter_type, fc, q, gain_db)
# x.shape is (n_channels, n_samples) or (n_samples, )
y = pyquad.filter(x)
```

### Online (frame by frame proc.)
```python
pyquad = PyQuadFilter(sr)
pyquad.prepare_filter_online(n_ch=n_ch, length=length)

# in callback func.
# x_frame.shape and y_frame.shape are (n_channels, length) or (length, )
pyquad.set_params(filter_type, fc, q, gain_db)
y_frame[:] = pyquad.filter(x_frame, online=True)
```

## Filter types
- "lowpass"
- "highpass"
- "bandpass"
- "allpass"
- "notch"
- "peaking"
- "lowshelf"
- "highshelf"

# Frequency response
Change cut-off freq.
https://github.com/Kurene/pyquadfilter/blob/main/plot_responses_fc/README.md

Change q-value:
https://github.com/Kurene/pyquadfilter/blob/main/plot_responses_q/README.md


# License
pyquadfilter is copyright [Kurene](https://twitter.com/_kurene), and licensed under
the MIT license.  I am providing code in this repository to you under an open
source license.  This is my personal repository; the license you receive to
my code is from me and not from my employer. See the `LICENSE` file for details.
