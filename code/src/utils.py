import numpy as np
from PIL import Image

def load_image(path):
    img = Image.open(path).convert("RGB").resize((512, 512))
    return np.asarray(img) / 255.0

def save_image(arr, path):
    img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def resize_np(arr, size):
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img).astype(float) / 255.0