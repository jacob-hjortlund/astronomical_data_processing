import os
import paths
import numpy as np
import ccdproc as ccdp
import matplotlib.pyplot as plt
import processing_utils as utils

from astropy import units as u

DIM = 1030
filter_name_column = "eso ins filt1 name"
raw_calib_path = paths.data / "raw_photometry" / "CALIB"
master_bias_path = (
    paths.data / "processed_photometry" / "calibration" / "bias" / "master_bias_40.fits"
)
# master_dark_path = (
#     paths.data / "processed_photometry" / "calibration" / "darks" / "master_dark.fits"
# )


# calib_collection = ccdp.ImageFileCollection(location=raw_calib_path)
master_bias = ccdp.CCDData.read(master_bias_path)
# master_dark = ccdp.CCDData.read(master_dark_path)

for flat_type in ["SKY", "LAMP"]:
    # flat_filter = {
    #     "object": flat_type + "FLAT",
    #     "naxis1": DIM,
    #     "naxis2": DIM,
    # }

    # flat_collection = calib_collection.filter(**flat_filter)
    raw_flat_path = raw_calib_path / (flat_type + "FLAT")
    flat_collection = ccdp.ImageFileCollection(location=raw_flat_path)
    filter_names = {
        filter_name[0]: filter_name
        for filter_name in flat_collection.summary.to_pandas()[
            filter_name_column
        ].unique()
    }

    for filter_name in filter_names.keys():
        filter_save_path = save_path = (
            paths.data
            / "processed_photometry"
            / "calibration"
            / "flats"
            / flat_type.lower()
            / filter_name
        )
        filter_name = filter_names[filter_name]
        filter_collection = flat_collection.filter(**{filter_name_column: filter_name})
        flats = list(filter_collection.ccds(ccd_kwargs={"unit": "adu"}))
        master_flat = utils.create_master(
            flats,
            image_type=flat_type + "FLAT",
            save=True,
            save_path=filter_save_path,
            master_bias=master_bias,
            # master_dark=master_dark,
        )
