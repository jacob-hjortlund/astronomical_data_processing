import paths
import os
import numpy as np
import ccdproc as ccdp
import processing_utils as utils
import shutil 

DIM = 1030
raw_calib_path = paths.data / "raw_photometry" / "CALIB"
calib_collection = ccdp.ImageFileCollection(location=raw_calib_path)

bias_files=calib_collection.files_filtered(NAXIS1=1030,NAXIS2=1030,OBJECT ='BIAS')
dark_files=calib_collection.files_filtered(NAXIS1=1030,NAXIS2=1030,OBJECT ='DARK')
skyflat_files=calib_collection.files_filtered(NAXIS1=1030,NAXIS2=1030,OBJECT ='SKYFLAT')
lampflat_files=calib_collection.files_filtered(NAXIS1=1030,NAXIS2=1030,OBJECT ='LAMPFLAT')


###### SORT THE BIAS FILES TO THEIR OWN DIR ######
if not os.path.exists(f'{paths.data}/raw_photometry/BIAS'):
    os.makedirs(f'{paths.data}/raw_photometry/BIAS')

# Copy the bias files from CALIB to the new BIAS
for bias_file in bias_files:
    shutil.copy2(f'{paths.data}/raw_photometry/CALIB/{bias_file}', f'{paths.data}/raw_photometry/BIAS/{bias_file}')


###### SORT THE DARK FILES TO THEIR OWN DIR ######
if not os.path.exists(f'{paths.data}/raw_photometry/DARK'):
    os.makedirs(f'{paths.data}/raw_photometry/DARK')

# Copy the dark files from CALIB to the new DARK
for dark_file in dark_files:
    shutil.copy2(f'{paths.data}/raw_photometry/CALIB/{dark_file}', f'{paths.data}/raw_photometry/DARK/{dark_file}')


###### SORT THE FLAT FILES TO THEIR OWN DIR ######
if not os.path.exists(f'{paths.data}/raw_photometry/FLAT'):
    os.makedirs(f'{paths.data}/raw_photometry/FLAT')
    
if not os.path.exists(f'{paths.data}/raw_photometry/FLAT/SKYFLAT'):
    os.makedirs(f'{paths.data}/raw_photometry/FLAT/SKYFLAT')
    os.makedirs(f'{paths.data}/raw_photometry/FLAT/SKYFLAT/R')
    os.makedirs(f'{paths.data}/raw_photometry/FLAT/SKYFLAT/V')
    os.makedirs(f'{paths.data}/raw_photometry/FLAT/SKYFLAT/B')
    
if not os.path.exists(f'{paths.data}/raw_photometry/FLAT/LAMPFLAT'):
    os.makedirs(f'{paths.data}/raw_photometry/FLAT/LAMPFLAT') 
    os.makedirs(f'{paths.data}/raw_photometry/FLAT/LAMPFLAT/R')
    os.makedirs(f'{paths.data}/raw_photometry/FLAT/LAMPFLAT/V')
    os.makedirs(f'{paths.data}/raw_photometry/FLAT/LAMPFLAT/B')
    

# Copy the bias files from CALIB to the new BIAS
for flat_file in skyflat_files:
    shutil.copy2(f'{paths.data}/raw_photometry/CALIB/{flat_file}', f'{paths.data}/raw_photometry/FLAT/SKYFLAT/{flat_file}')
for flat_file in lampflat_files:
    shutil.copy2(f'{paths.data}/raw_photometry/CALIB/{flat_file}', f'{paths.data}/raw_photometry/FLAT/LAMPFLAT/{flat_file}')

