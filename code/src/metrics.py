import numpy as np

def mse(a, b):
    return np.mean((a - b)**2)

def psnr(mse):
    return 10 * np.log10(1.0 / mse)