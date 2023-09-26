import os
import paths
import utils
import numpy as np
import ccdproc as ccdp
import matplotlib.pyplot as plt

from astropy import units as u

DIM = 1030
raw_calib_path = paths.data / "raw_photometry" / "CALIB"
save_path = paths.data / "processed_photometry" / "calibration" / "darks"
master_bias_path = (
    paths.data / "processed_photometry" / "calibration" / "bias" / "master_bias_40.fits"
)

calib_collection = ccdp.ImageFileCollection(location=raw_calib_path)
master_bias = ccdp.CCDData.read(master_bias_path, unit="adu")

dark_filter = {
    "object": "DARK",
    "naxis1": DIM,
    "naxis2": DIM,
}

dark_collection = calib_collection.filter(**dark_filter)
darks = list(dark_collection.ccds(ccd_kwargs={"unit": "adu"}))

master_dark = utils.create_master(
    darks, image_type="DARK", save=True, save_path=save_path, master_bias=master_bias
)
