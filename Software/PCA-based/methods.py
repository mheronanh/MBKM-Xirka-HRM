#%% Library
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from scipy.signal import find_peaks
#%%Functions
def PCA_method(s1,s2,s3):
    InputData = list(zip(s1,s2,s3))
    InputData_processed = pd.DataFrame(data=InputData)
    X = StandardScaler().fit_transform(InputData_processed)
    pca = PCA(n_components=3)
    PCA_values = pca.fit_transform(X)

 #Converting PCA Component (type NumPy Array to Pandas DataFrame)
    PCA_df = pd.DataFrame(data=PCA_values,columns=['Component 1','Component 2','Component 3'])
    PCA_value1 = PCA_df['Component 1'].tolist()
    PCA_value2 = PCA_df['Component 2'].tolist()
    PCA_value3 = PCA_df['Component 3'].tolist()
    return PCA_value1,PCA_value2,PCA_value3

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