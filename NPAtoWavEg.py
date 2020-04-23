import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


samplerate = 44100; fs = 100
t = np.linspace(0., 1., samplerate)
amplitude = np.iinfo(np.int16).max
data = amplitude * np.sin(2. * np.pi * fs * t)
plt.plot(data)
plt.show()
# write("example.wav", samplerate, data)
