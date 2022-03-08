#%% Library
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import numpy as np
from scipy.signal import find_peaks
#%%Functions
def ICA_method(s1,s2,s3):
    InputData = list(zip(s1,s2,s3))
    InputData_processed = pd.DataFrame(data=InputData)
    X = StandardScaler().fit_transform(InputData_processed)
    ica = FastICA(n_components=3,max_iter=2000)
    ICA_values = ica.fit_transform(X)

 #Converting PCA Component (type NumPy Array to Pandas DataFrame)
    ICA_df = pd.DataFrame(data=ICA_values,columns=['Component 1','Component 2','Component 3'])
    ICA_value1 = ICA_df['Component 1'].tolist()
    ICA_value2 = ICA_df['Component 2'].tolist()
    ICA_value3 = ICA_df['Component 3'].tolist()
    return ICA_value1,ICA_value2,ICA_value3

def find_MAX(ps):
    peaks = find_peaks(ps,height=0)
    tinggi = peaks[1]['peak_heights']
    tinggi.sort()
    #Calculating two highest peak to find periodicity
    max1 = tinggi[len(tinggi)-1]
    max2 = tinggi[len(tinggi)-2]
    return max1/max2

def choose_component(s1,s2,s3):
    #Initialize variable
    list_max = []
    
    #Calculating power spectrum
    ps1 = np.abs(np.fft.rfft(s1))**2
    ps2 = np.abs(np.fft.rfft(s2))**2
    ps3 = np.abs(np.fft.rfft(s3))**2
    
    #Finding maximum
    max1 = find_MAX(ps1)
    max2 = find_MAX(ps2)
    max3 = find_MAX(ps3)
    list_max.append(max1)
    list_max.append(max2)
    list_max.append(max3)
    list_max.sort()
    if (list_max[2]==max1):
        return s1
    elif (list_max[2]==max2):
        return s2
    else:
        return s3