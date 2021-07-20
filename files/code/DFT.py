import numpy as np

def DFT(M):
    M = np.array(M)
    try:
        N, _ = M.shape
    except:
        N, = M.shape
    FM = np.array(range(N))
    FM = FM.reshape(-1, 1).dot(FM.reshape(1, -1))
    FM = np.exp(FM * (-2 * np.pi * 1j / N))
    print('Discrete Fourier Transform Matrix:\n', FM)
    return FM.dot(M)

def IDFT(M):
    M = np.array(M)
    try:
        N, _ = M.shape
    except:
        N, = M.shape
    IFM = np.array(range(N))
    IFM = IFM.reshape(-1, 1).dot(IFM.reshape(1, -1))
    IFM = np.exp(IFM * (2 * np.pi * 1j / N))
    IFM = IFM /  N
    print('Inverse Discrete Fourier Transform Matrix:\n', IFM)
    return IFM.dot(M)
    