#%% Library
import numpy as np
import preprocessing as prep
from scipy.signal import convolve
#%% Functions
def cut_window (array,awal,akhir):
    panjang = akhir-awal
    temp = [None]*panjang; 
    i = 0
    j = awal
    for i in range(i,panjang) :
        temp[i]=array[j]
        i+=1
        j+=1
    return temp

def calculate_bpm(s,fs,lowcut,highcut,window):
    list_bpm  = []

    #Time Step
    time_step = 1/fs
    taps = prep.bandpass_firwin(128, lowcut, highcut, fs)
    i = 0
    j= window-1   
    
    while (j<= len(s)):
        #Windowing
        temp = cut_window(s,i,j)
        
        #Calculate bpm
        filtered = convolve(temp, taps, mode='full')
        ps = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.fftfreq(len(filtered), d=time_step)
        index_f_value = np.argmax(ps)
        f_value = freqs[index_f_value]
        bpm = 60*f_value
        list_bpm.append(bpm)
        i+=fs
        j+=fs
    return list_bpm
