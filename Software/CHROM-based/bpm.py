#%% Library
import numpy as np
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

def calculate_bpm(s,fs,window):
    list_bpm  = []
    
    #Time Step
    time_step = 1/fs
    i = 0
    j= window-1   
    
    while (j<= len(s)):
        #Windowing
        temp = cut_window(s,i,j)
        
        #Calculate bpm
        ps = np.abs(np.fft.rfft(temp))
        freqs = np.fft.fftfreq(len(temp), d=time_step)
        index_f_value = np.argmax(ps)
        f_value = freqs[index_f_value]
        bpm = 60*f_value
        list_bpm.append(bpm)
        i+=fs
        j+=fs
    return list_bpm

def thresholding_bpm(s1,s2):
    for i in range (1,len(s1)):
        if (abs(s1[i-1]-s1[i])>12):
            if (abs(s1[i-1]-s2[i])<=12):
                s1[i]=s2[i]
            else:
                s1[i]=s1[i-1]
    return s1