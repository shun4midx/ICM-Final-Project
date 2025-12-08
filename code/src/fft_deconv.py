import numpy as np

def pad_kernel(K, shape):
    pad = np.zeros(shape)
    kh, kw = K.shape
    pad[:kh, :kw] = K
    return pad

def apply_kernel_fft(x, K):
    # x: H x W x 3
    # K: either (kh, kw)  or (3, kh, kw)

    H, W, _ = x.shape

    # Case 1: shared kernel across RGB
    if K.ndim == 2:
        Khat = np.fft.fft2(pad_kernel(K, (H, W)))
        y = np.zeros_like(x)
        for c in range(3):
            Xhat = np.fft.fft2(x[:, :, c])
            Yhat = Khat * Xhat
            y[:, :, c] = np.real(np.fft.ifft2(Yhat))
        return y

    # Case 2: per-channel RGB kernel
    elif K.ndim == 3:
        y = np.zeros_like(x)
        for c in range(3):
            Kc = K[c]
            Khat = np.fft.fft2(pad_kernel(Kc, (H, W)))
            Xhat = np.fft.fft2(x[:, :, c])
            Yhat = Khat * Xhat
            y[:, :, c] = np.real(np.fft.ifft2(Yhat))
        return y

    else:
        raise ValueError("Kernel K must be 2D or 3D")

def apply_kernel_fft_rgb(x, K):
    # K has shape (3, kH, kW)
    out = np.zeros_like(x)
    
    for c in range(3):
        out[:, :, c] = apply_kernel_fft(x[:, :, c], K[c])
    
    return out

def deconv_fft(z, K, lam):
    H, W, _ = z.shape
    xhat = np.zeros_like(z)

    # Shared kernel
    if K.ndim == 2:
        Khat = np.fft.fft2(pad_kernel(K, (H,W)))
        denom = np.abs(Khat)**2 + lam
        for c in range(3):
            Zhat = np.fft.fft2(z[:, :, c])
            Xhat = (np.conj(Khat) * Zhat) / denom
            xhat[:, :, c] = np.real(np.fft.ifft2(Xhat))
        return xhat

    # RGB kernel
    elif K.ndim == 3:
        for c in range(3):
            Kc = K[c]
            Khat = np.fft.fft2(pad_kernel(Kc, (H,W)))
            denom = np.abs(Khat)**2 + lam
            Zhat = np.fft.fft2(z[:, :, c])
            Xhat = (np.conj(Khat) * Zhat) / denom
            xhat[:, :, c] = np.real(np.fft.ifft2(Xhat))
        return xhat

    else:
        raise ValueError("Kernel K must be 2D or 3D")

def deconv_fft_rgb(z, K, lam):
    out = np.zeros_like(z)

    for c in range(3):
        out[:, :, c] = deconv_fft(z[:, :, c], K[c], lam)

    return out