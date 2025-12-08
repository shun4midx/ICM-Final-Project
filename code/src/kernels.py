import numpy as np
from scipy.ndimage import rotate
from utils import load_image, resize_np

def gaussian_kernel(size=11, sigma=2):
    ax = np.arange(-(size//2), size//2+1)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2) / (2*sigma*sigma))
    return g / g.sum()

def box_kernel(size=9):
    return np.ones((size, size)) / (size*size)

def motion_kernel(length=15, angle=0):
    k = np.zeros((length, length))
    k[length//2, :] = 1
    k = rotate(k, angle, reshape=False)
    return k / k.sum()

def random_kernel(size=15, seed=123):
    np.random.seed(seed)
    k = np.random.rand(size, size)
    return k / k.sum()

def key_image_kernel_rgb(path, size=15):
    # Load RGB key image as float array
    img = load_image(path)  # returns 512x512x3 normalized
    
    # Extract RGB channels
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Resize channels
    KR = resize_np(R, size)
    KG = resize_np(G, size)
    KB = resize_np(B, size)

    # Normalize each channel to sum = 1
    KR = KR / np.sum(KR)
    KG = KG / np.sum(KG)
    KB = KB / np.sum(KB)

    # Return RGB kernel (3, size, size)
    return np.stack([KR, KG, KB], axis=0)