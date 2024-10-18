import numpy as np

def clfft(x, y, fft_type, res=None, zpad=30):
    """
    Parameters:
    ----------
    x : array-like
        Time or frequency vector.
    y : array-like
        Signal vector, y(t, n) where n is a parameter.
    fft_type : str
        'tf' for time to frequency or 'ft' for frequency to time.
    res : int, optional
        Resolution of the FFT, default is the next power of 2 based on signal length.
    zpad : int or str, optional
        Zeropadding method with decay rate of exponential tail (default 30), or 'nopadding'.

    Returns:
    -------
    f : array
        Frequency or time vector.
    F : array
        FFT of the input signal.
    absF : array
        Magnitude of the FFT.
    angF : array
        Unwrapped phase of the FFT.
    """
    
    # Get the shape of y

    y = np.atleast_2d(y)
    a = y.shape
    print(a)
    
    # If res not set, find nearest power of 2
    if res is None:
        res = 2**int(np.ceil(np.log2(max(a))))
    
    # If zpad is not set, default to 30
    if zpad is None:
        zpad = 30
    
    # If res is odd, make it even
    if res % 2 != 0:
        res += 1
    
    # Transpose if needed (to ensure correct shape)
    if a[0] < a[1]:
        y = y.T
    
    # Zeropadding with exponential tail
    if res > max(a) and zpad != 'nopadding':
        zpl = (res - max(a)) // 2  # Half of zero-padding length
        ap = zpad
        y = np.vstack([
            np.exp(-ap / zpl * np.arange(zpl, 0, -1))[:, None] * y[0, :],
            y,
            np.exp(-ap / zpl * np.arange(1, zpl+1))[:, None] * y[-1, :]
        ])
    
    # Calculate frequency or time axis
    df = len(x) / ((max(x) - min(x)) * res)
    f = np.linspace(-res / 2 * df, (res / 2 - 1) * df, res)
    
    # FFT operations based on type
    if fft_type == 'tf':  # Time to frequency
        if min(a) == 1:
            F = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(y), res))
            angF = np.unwrap(np.angle(F))
        else:
            F = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(y, axes=0), res, axis=0), axes=0)
            angF = np.unwrap(np.unwrap(np.angle(F), axis=0), axis=1)
    
    elif fft_type == 'ft':  # Frequency to time
        if min(a) == 1:
            F = np.fft.fftshift(np.fft.fft(np.fft.fftshift(y), res))
            angF = np.unwrap(np.angle(F))
        else:
            F = np.fft.fftshift(np.fft.fft(np.fft.fftshift(y, axes=0), res, axis=0), axes=0)
            angF = np.unwrap(np.unwrap(np.angle(F), axis=0), axis=1)
    
    absF = np.abs(F)
    
    return f, F, absF, angF
