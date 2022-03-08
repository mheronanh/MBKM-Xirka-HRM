#%%Note
#Tested on VIPL-HR Dataset
#Make sure laptop is plugged in when testing

#%% Library
import csv_reader as csv_read
import vid_reader as vid
import preprocessing as prep
import methods as method
import bpm as bpm
import cv2
import numpy as np
#%% Main Function
def main(i):
    # Parameters
    lowcut              = 0.75
    highcut             = 4
    video_path          = r'D:\ITB\Semester 7\Dataset VIPL-HR\p{}\v1\source3\video.avi'.format(i)
    video_cap           = cv2.VideoCapture(video_path)
    fs                  = video_cap.get(cv2.CAP_PROP_FPS) 
    window              = 400
    data_amount         = 600 #in frames
    ref_path            = r'D:\ITB\Semester 7\Dataset VIPL-HR\p{}\v1\source3\gt_HR.csv'.format(i)
    ref                 = csv_read.get_hr_from_csv(ref_path)
    
    #Preparing variable
    red_channel = []
    green_channel = []
    blue_channel = []
    duration = 0

    #Collect data from video
    red_channel,green_channel,blue_channel, duration,fs = vid.read_video(video_path,data_amount)
    
    #%%Signal Preprocessing
    det_r,det_g,det_b = prep.detrend_signal(red_channel,green_channel,blue_channel)
    
    norm_r,norm_g,norm_b = prep.normalize_signal(det_r,det_g,det_b)
    
    smo_r,smo_g,smo_b = prep.moving_average(norm_r, norm_g,norm_b)
    
    #RPPG Method 
    
    #PCA method
    ppg1,ppg2,ppg3 = method.PCA_method(smo_r,smo_g,smo_b)
    
    ppg_final = method.choose_component(ppg1,ppg2,ppg3)
    
    
    list_bpm_final = bpm.calculate_bpm(ppg_final, fs, lowcut, highcut,window)
    
    print(list_bpm_final)
    print("Rata-Rata: "+ str(np.mean(list_bpm_final)))
    print("ref: "+ str(ref))
    print("durasi: "+str(duration))
    return (abs((np.mean(list_bpm_final))-ref))

#%% Main Program
arr_diff =[]
for i in range(102,108):
    print("Video saat ini= p"+str(i))
    diff = main(i)
    arr_diff.append(diff)
    csv_read.write_to_csv(arr_diff)