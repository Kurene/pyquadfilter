# PyQuadFilter

Implementation of digital bi-quad filter in Python

reference:
https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html


# Quick Start

## install
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

```python
pyquad = PyQuadFilter(sr)
pyquad.set_params(filter_type, fc, q, gain_db)
# x.shape is (n_channels, n_samples) or (n_samples, )
y = pyquad.filter(x)
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

# License
pyquadfilter is copyright [Kurene](https://twitter.com/_kurene), and licensed under
the MIT license.  I am providing code in this repository to you under an open
source license.  This is my personal repository; the license you receive to
my code is from me and not from my employer. See the `LICENSE` file for details.
