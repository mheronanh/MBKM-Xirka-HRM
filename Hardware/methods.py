#%% Library
import numpy as np
import preprocessing as prep
from scipy.signal import convolve
#%%Functions

def CHROM_method(s1,s2,s3,fs):
        # CHROM
    X_chrom     = np.array(s1) - np.array(s2)
    Y_chrom     = np.array(s1) + np.array(s2) - 2 * np.array(s3)
    taps        = prep.bandpass_firwin(128, 1,3, fs)
    Xf          = convolve(X_chrom, taps, mode='full')
    Yf          = convolve(Y_chrom, taps, mode='full')
    Nx          = np.std(Xf)
    Ny          = np.std(Yf)
    alpha_chrom = Nx/Ny
    fin_chrom   = Xf - alpha_chrom * Yf

    return fin_chrom