#%% Library
import pandas as pd
import csv
#%% Functions
def get_hr_from_csv (path): 
    data = pd.read_csv(path)
    hr_average = data['HR'].mean()
    
    return hr_average  

def write_to_csv(arr):
    csv_file =  open(r"D:\ITB\Semester 7\Hasil Pengujian\Paper\CHROM\v1\CHROM1.csv", 'w')
    csv_writer = csv.writer(csv_file, delimiter=",",lineterminator = '\n')
    csv_writer.writerows(map(lambda x: [x], arr))
    csv_file.close()