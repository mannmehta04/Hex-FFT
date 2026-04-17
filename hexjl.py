import numpy as np

def pad_zeros(arr):
    return np.hstack((arr, np.zeros_like(arr)))

def decimate(data, offset):
    return data[:, offset::2]

def div2(n):
    return n >> 1

def fold_half(data, op):
    n = data.shape[0]
    return op(data[:div2(n), :], data[div2(n):n, :])

def dft_nst1(data):
    out = np.fft.fft(pad_zeros(data), axis=1)
    return decimate(out, 0), decimate(out, 1)

def nst2(data):
    folded = fold_half(data, np.add)
    ffted = np.fft.fft(folded, axis=0)
    return np.tile(ffted, (2, 1))

def coeff_inner(i, n):
    return np.exp(-2j * np.pi * i / n)

def nst3_coeffs(n, m):
    return coeff_inner(np.arange(div2(n)), n).reshape(div2(n), 1)

def nst3(data):
    n, m = data.shape
    coeffs = nst3_coeffs(n, m)
    folded = fold_half(data, np.subtract)
    ffted = np.fft.fft(coeffs * folded, axis=0)
    return np.tile(ffted, (2, 1))

def w_inner(b, s, d, n, m):
    return np.exp(-1j * np.pi * ((b + 2*d) / (2*m) + (b + 2*s) / n))

def w(b, n, m):
    return np.array([[w_inner(b, s, d, n, m) for d in range(m)] for s in range(n)])

def hfft2(data0, data1):
    g00, g01 = dft_nst1(data0)
    g10, g11 = dft_nst1(data1)
    return nst2(g00) + w(0, *g10.shape) * nst2(g10), nst3(g01) + w(1, *g11.shape) * nst3(g11)

def idft_inst1(data):
    out = 2 * np.fft.ifft(pad_zeros(data), axis=1)
    return decimate(out, 0), decimate(out, 1)

def inst2(data):
    folded = fold_half(data, np.add)
    iffted = 0.5 * np.fft.ifft(folded, axis=0)
    return np.tile(iffted, (2, 1))

def icoeff_inner(i, n):
    return np.exp(2j * np.pi * i / n)

def inst3_coeffs(n, m):
    return icoeff_inner(np.arange(div2(n)), n).reshape(div2(n), 1)

def inst3(data):
    n, m = data.shape
    coeffs = inst3_coeffs(n, m)
    folded = fold_half(data, np.subtract)
    iffted = 0.5 * np.fft.ifft(coeffs * folded, axis=0)
    return np.tile(iffted, (2, 1))

def iw_inner(a, r, c, n, m):
    return np.exp(1j * np.pi * ((a + 2*c) / (2*m) + (a + 2*r) / n))

def iw(a, n, m):
    return np.array([[iw_inner(a, r, c, n, m) for c in range(m)] for r in range(n)])

def ihfft2(data0, data1):
    g00, g01 = idft_inst1(data0)
    g10, g11 = idft_inst1(data1)
    return 0.5 * (inst2(g00) + iw(0, *g10.shape) * inst2(g10)), 0.5 * (inst3(g01) + iw(1, *g11.shape) * inst3(g11))

class OffsetHexData:
    def __init__(self, odd_rows, even_rows):
        assert odd_rows.shape == even_rows.shape
        self.odd_rows = odd_rows
        self.even_rows = even_rows

    def fft(self):
        odd_fft, even_fft = hfft2(self.odd_rows, self.even_rows)
        return OffsetHexData(odd_fft, even_fft)

    def ifft(self):
        odd_ifft, even_ifft = ihfft2(self.odd_rows, self.even_rows)
        return OffsetHexData(odd_ifft, even_ifft)