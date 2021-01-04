"""
Python implementation of digital biquad filters

reference: https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
"""
import numpy as np
from scipy import signal
from numba import jit
from .plot import plot_frequency_response

@jit
def biquad_filter(length, input, output, b, a):
    """
    a[0] == 1.0
    length == len(input) - 4
    """
    for k in range(length):
        output[k] = b[0] * input[k]\
                  + b[1] * input[k-1]\
                  + b[2] * input[k-2]\
                  - a[1] * output[k-1]\
                  - a[2] * output[k-2]          
        

class PyQuadFilter():
    """ Bi-quad filter on python

    Parameters
    ----------
    rate : float
        Sampling rate  (> 0.0)
    filter_class : str
        - "lowpass"
        - "highpass"
        - "bandpass"
        - "allpass"
        - "notch"
        - "peaking"
        - "lowshelf"
        - "highshelf"
    fc : float
        Cut-off frequency (> 0.0)
    q : float
        q value (> 0.0)
    gain_db : float
        gain in dB
    """
    def __init__(
        self, 
        sr, 
        filter_type=None,
        fc=1000,
        q=1.0,
        gain_db=1.0,
    ):
        if sr < 0.0:
            raise ValueError("sr cannot be given a negative value")
        self._sr = sr
    
        self._b, self._a = np.zeros(3), np.zeros(3)
        
        if filter_type is not None:
            self.set_params(filter_type, fc, q, gain_db)


    def set_params(self, filter_type, fc, q, gain_db):
        b, a = self._b.view(), self._a.view()
        b *= 0.0
        a *= 0.0
    
        if fc < 0.0 or fc >= self._sr / 2.0:
            raise ValueError(f"illegal value: fc={fc}")
        self._fc = fc

        if q < 0.0:
            raise ValueError(f"illegal value: q={q}")
        self._q = q
        
        self._gain_db = gain_db
        self.amp = 10.0 ** (gain_db/40.0)
        
        w = 2.0 * np.pi * self._fc / self._sr
        cos_w, sin_w = np.cos(w), np.sin(w)
        alpha = 0.5 * sin_w / self._q
        
        filter_type = filter_type.lower()\
                        .replace("-","").replace("_","")
        if filter_type == "lowpass":
            b[0] = (1.0 - cos_w) * 0.5
            b[1] = (1.0 - cos_w)
            b[2] = (1.0 - cos_w) * 0.5
            a[0] = 1.0 + alpha
            a[1] = -2.0 * cos_w
            a[2] = 1.0 - alpha
        elif filter_type == "highpass":
            b[0] = (1.0 + cos_w) * 0.5
            b[1] = -(1.0 + cos_w)
            b[2] = (1.0 + cos_w) * 0.5
            a[0] = 1.0 + alpha
            a[1] = -2.0 * cos_w
            a[2] = 1.0 - alpha
        elif filter_type == "bandpass":
            b[0] =  sin_w * 0.5
            b[1] = 0.0
            b[2] = -sin_w * 0.5
            a[0] = 1.0 + alpha
            a[1] = -2.0 * cos_w
            a[2] = 1.0 - alpha
        elif filter_type == "allpass":
            b[0] = 1.0 - alpha
            b[1] = -2.0 * cos_w
            b[2] = 1.0 + alpha
            a[0] = 1.0 + alpha
            a[1] = -2.0 * cos_w
            a[2] = 1.0 - alpha
        elif filter_type == "notch":
            b[0] = 1.0
            b[1] = -2.0 * cos_w
            b[2] = 1.0
            a[0] = 1.0 + alpha
            a[1] = -2.0 * cos_w
            a[2] = 1.0 - alpha
        elif filter_type == "peaking":
            b[0] = 1.0 + alpha * self.amp
            b[1] = -2.0 * cos_w
            b[2] = 1.0 - alpha * self.amp
            with np.errstate(divide='ignore'):
                a[0] = 1.0 + alpha / self.amp
                a[1] = -2.0 * cos_w
                a[2] = 1.0 - alpha / self.amp
        elif filter_type == "lowshelf":
            amp_add_1 = self.amp + 1.0
            amp_sub_1 = self.amp - 1.0
            sqrt_amp  = np.sqrt(self.amp)
            b[0] = self.amp * (amp_add_1 - amp_sub_1 * cos_w\
                 + 2.0 * sqrt_amp * alpha)
            b[1] = 2.0 * self.amp * (amp_sub_1 - amp_add_1 * cos_w)
            b[2] = self.amp * (amp_add_1 - amp_sub_1 * cos_w\
                 - 2.0 * sqrt_amp * alpha)
            a[0] = amp_add_1 + amp_sub_1 * cos_w\
                 + 2.0 * sqrt_amp * alpha
            a[1] = -2.0 * (amp_sub_1 + amp_add_1 * cos_w)
            a[2] = amp_add_1 + amp_sub_1 * cos_w\
                 - 2.0 * sqrt_amp * alpha
        elif filter_type == "highshelf":
            amp_add_1 = self.amp + 1.0
            amp_sub_1 = self.amp - 1.0
            sqrt_amp  = np.sqrt(self.amp)
            
            b[0] = self.amp * (amp_add_1 + amp_sub_1 * cos_w\
                 + 2.0 * sqrt_amp * alpha)
            b[1] = -2.0 * self.amp * (amp_sub_1 + amp_add_1 * cos_w)
            b[2] = self.amp * (amp_add_1 + amp_sub_1 * cos_w\
                 - 2.0 * sqrt_amp * alpha)
            a[0] = amp_add_1 - amp_sub_1 * cos_w\
                 + 2.0 * sqrt_amp * alpha
            a[1] = 2.0 * (amp_sub_1 - amp_add_1 * cos_w)
            a[2] = amp_add_1 - amp_sub_1 * cos_w\
                 - 2.0 * sqrt_amp * alpha
        else:
            raise ValueError(f"invalid filter_type: {filter_type}")
            
        self.filter_type = filter_type
        b /= a[0]
        a /= a[0]
        
        return b, a

    def filter(self, x):
        y = np.zeros(x.shape)
        if x.ndim == 1:
            length = x.shape[0] + 4
            x_tmp = np.zeros(length)
            y_tmp = np.zeros(length)
            x_tmp[2:length-2] = x
            biquad_filter(length, x_tmp, y_tmp, self._b, self._a)
            y[:] = y_tmp[2:length-2]
        elif x.ndim == 2:
            length = x.shape[1] + 4
            n_ch = x.shape[0]
            x_tmp = np.zeros(length)
            y_tmp = np.zeros(length)
            for k in range(n_ch):
                x_tmp[2:length-2] = x[k,:]
                biquad_filter(length, x_tmp, y_tmp, self._b, self._a)
                y[k,:] = y_tmp[2:length-2]
                
        return y
    
    def plot_frequency_response(self, **kwargs):
        plot_frequency_response(self._sr, self._b, self._a, **kwargs)
    
    def __str__(self):
        ret = f"""
        type:    {self.filter_type}
        fc:      {self._fc} Hz
        q:       {self._q}
        gain_db: {self._gain_db}
        
        b: {self._b}
        a: {self._a}
        """
        return ret

    @property
    def b(self):
        return self._b

    @property
    def a(self):
        return self._a

    @property
    def coefs(self):
        return self._b, self._a

    @property
    def sr(self):
        return self._sr

    @property
    def q(self):
        return self._q
        
    @property
    def fc(self):
        return self._fc

    @property
    def gain_db(self):
        return self._gain_db
