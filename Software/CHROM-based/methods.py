#%% Library
import numpy as np
import preprocessing as prep
from scipy.signal import convolve
#%%Functions

def CHROM_method(s1,s2,s3,low_cut,high_cut,fs):
        # CHROM
    X_chrom     = s1 - s2
    Y_chrom     = s1 + s2 - 2 * s3
    taps        = prep.bandpass_firwin(128, low_cut, high_cut, fs)
    Xf          = convolve(X_chrom, taps, mode='full')
    Yf          = convolve(Y_chrom, taps, mode='full')
    Nx          = np.std(Xf)
    Ny          = np.std(Yf)
    alpha_chrom = Nx/Ny
    fin_chrom   = Xf - alpha_chrom * Yf

    return fin_chrom
