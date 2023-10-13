import os
import paths
import numpy as np
import ccdproc as ccdp
import processing_utils as utils
import matplotlib.pyplot as plt
import astropy

BIAS_path = paths.data / "raw_photometry" / "BIAS"
# Create an empty list to store the CCDData objects for bias frames
bias_frames = []
stats = []

fig = plt.figure()
plt.xlabel('Pixel Value (ADU)')
plt.ylabel('Frequency')
plt.title('Histogram of Bias Images Pixel Values 100-200 [ADU]')

# Load bias frames and store them in the list
for file in os.listdir(BIAS_path):
    bias_data = ccdp.CCDData.read(f'{BIAS_path}/{file}', unit='adu')
    bias_frames.append(bias_data)

    plt.hist(bias_data.data.flatten(), bins=100, range=(100,200), color='b', alpha=1/40)
    
    region0= bias_data[21:1019, 21:1019]
    
    region1= bias_data[21:510, 21:510]
    region2= bias_data[21:510, 510:1020]
    region3= bias_data[510:1020, 21:510]
    region4= bias_data[510:1020, 510:1020]
    
    def thestats(region):
        min_val = np.nanmin(region)
        max_val = np.nanmax(region)
        mean_val = np.nanmean(region)
        std_val = np.nanstd(region) 
        return min_val, max_val, mean_val, std_val
    
    min_val0, max_val0, mean_val0, std_val0 =thestats(region0)
    min_val1, max_val1, mean_val1, std_val1 =thestats(region1)
    min_val2, max_val2, mean_val2, std_val2 =thestats(region2)
    min_val3, max_val3, mean_val3, std_val3 =thestats(region3)
    min_val4, max_val4, mean_val4, std_val4 =thestats(region4)
    
    # Modify the appending of statistics to the list
    stats.append([file,
              mean_val0, mean_val1, mean_val2, mean_val3, mean_val4,
              std_val0, std_val1, std_val2, std_val3, std_val4,
              min_val0, min_val1, min_val2, min_val3, min_val4,
              max_val0, max_val1, max_val2, max_val3, max_val4])

# Create an Astropy table from the list of statistics
stats_table = astropy.table.Table(rows=stats, names=('File Name', 
                                       'Mean0', 'Mean1', 'Mean2', 'Mean3', 'Mean4',
                                       'STD0', 'STD1', 'STD2', 'STD3', 'STD4',
                                       'Min0', 'Min1', 'Min2', 'Min3', 'Min4',
                                       'Max0', 'Max1', 'Max2', 'Max3', 'Max4'))
fig.savefig(
    paths.figures / "bias_all_histogram_2_1_3_images.pdf",
    bbox_inches="tight",
)