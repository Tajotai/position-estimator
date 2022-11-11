import numpy as np
import math
import matplotlib.pyplot as plt

beta = 4
raw = np.array([np.random.normal() for i in range(1000000)])
filt = np.arange(0,10000) ** (-beta / 2 + 1) / math.gamma(beta / 2)
sig = np.convolve(raw, filt, mode='valid')
S = np.fft.fft(sig)
# plt.plot(np.log(np.arange(0,10000)), np.log(np.abs(S))[:10000])
plt.plot(sig)
plt.show()