import numpy as np
from scipy.sparse import spdiags
import statistics
#%% Library
from scipy.signal import firwin,detrend
#%% Individual Functions

#Detrending
def advanced_detrending(input_data, Lambda):
    #ref : https://www.idiap.ch/software/bob/docs/bob/bob.rppg.base/v1.0.3/_modules/bob/rppg/cvpr14/filter_utils.html
  signal_length = len(input_data)

  # observation matrix
  H = np.identity(signal_length) 

  # second-order difference matrix
  ones = np.ones(signal_length)
  minus_twos = -2*np.ones(signal_length)
  diags_data = np.array([ones, minus_twos, ones])
  diags_index = np.array([0, 1, 2])
  D = spdiags(diags_data, diags_index, (signal_length-2), signal_length).toarray()
  filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda**2) * np.dot(D.T, D))), input_data)
  return filtered_signal
#Normalization
def normalize(s):
    normalized = (s -np.mean(s))/statistics.stdev(s)
    return normalized

#Moving average filter
def smooth_signal(unfiltered_signal):
    signal_length = len(unfiltered_signal)
    smooth_signal = []
    
    for i in range(signal_length):
        if (i <= len(unfiltered_signal) - 3) and (i >= 2):
            calc = (unfiltered_signal[i+2]+unfiltered_signal[i+1]+unfiltered_signal[i]+unfiltered_signal[i-1]+unfiltered_signal[i-2])/5
        elif (i == 1):
            calc = (unfiltered_signal[i+2]+unfiltered_signal[i+1]+unfiltered_signal[i]+unfiltered_signal[i-1])/4
        elif (i == 0):
            calc = (unfiltered_signal[i+2]+unfiltered_signal[i+1]+unfiltered_signal[i])/3
        elif (i == len(unfiltered_signal) - 2):
            calc = (unfiltered_signal[i+1]+unfiltered_signal[i]+unfiltered_signal[i-1]+unfiltered_signal[i-2])/4
        else:
            calc = (unfiltered_signal[i]+unfiltered_signal[i-1]+unfiltered_signal[i-2])/3
            
        smooth_signal.append(calc)
    
    return np.array(smooth_signal)

#Hamming bandpass filter
def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

#%% Combined Functions for all channels

def detrend_signal(s1,s2,s3):
    s1 = advanced_detrending(s1, 10)
    s2 = advanced_detrending(s2, 10)
    s3 = advanced_detrending(s3, 10)
    return s1,s2,s3

def detrend_signal2(s1,s2,s3):
    s1 = detrend(s1)
    s2 = detrend(s2)
    s3 = detrend(s3)
    return s1,s2,s3

def normalize_signal(s1,s2,s3):
    s1 = normalize(s1)
    s2 = normalize(s2)
    s3 = normalize(s3)
    return s1,s2,s3
def moving_average(s1,s2,s3):
    s1 = smooth_signal(s1)
    s2 = smooth_signal(s2)
    s3 = smooth_signal(s3)
    return s1,s2,s3