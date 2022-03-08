#%% Library
import numpy as np
import preprocessing as prep
import methods as method
import bpm as bpm
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

def calculate_bpm(s,fs):
    #Time Step
    time_step = 1/fs
    ps = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(len(s), d=time_step)
    index_f_value = np.argmax(ps)
    f_value = freqs[index_f_value]
    bpm = 60*f_value
    return bpm
   

def process_HR(red_channel,green_channel,blue_channel,fs):
    det_r,det_g,det_b = prep.detrend_signal(red_channel,green_channel,blue_channel)
    
    norm_r,norm_g,norm_b = prep.normalize_signal(det_r,det_g,det_b)
    
    smo_r,smo_g,smo_b = prep.moving_average(norm_r, norm_g,norm_b)
    
    #Chrominance Method
    ppg_final = method.CHROM_method(smo_r,smo_g,smo_b, fs)
    
    bpm_result = bpm.calculate_bpm(ppg_final,fs)
    return bpm_result