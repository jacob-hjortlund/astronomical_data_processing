import os
import paths
import utils
import numpy as np
import ccdproc as ccdp
import matplotlib.pyplot as plt

from astropy import units as u

DIM = 1030
raw_calib_path = paths.data / "raw_photometry" / "CALIB"
save_path = paths.data / "processed_photometry" / "CALIB" / "darks"
master_bias_path = (
    paths.data / "processed_photometry" / "CALIB" / "bias" / "master_BIAS_40.fits"
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

# fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
# ax.flatten()

# for i, dark in enumerate(darks):
#     if i < 4:
#         utils.show_image(dark, fig=fig, ax=ax[i // 2, i % 2], cbar_label="Signal [ADU]")
#         ax[i // 2, i % 2].set_title(f"Dark {i + 1}", fontsize=20)
#         ax[i // 2, i % 2].set_xlabel("X [pixels]", fontsize=16)
#         ax[i // 2, i % 2].set_ylabel("Y [pixels]", fontsize=16)

#     exposure_time = dark.header["EXPTIME"] * u.s
#     bias_subtracted_dark = ccdp.subtract_bias(dark, master_bias)
#     rescaled_dark = bias_subtracted_dark.multiply(1.0 / exposure_time)
#     rescaled_dark.write(save_path / f"calibrated_dark_{i}.fits", overwrite=True)

# fig.suptitle("Randomly Selected Dark Frames", fontsize=24)
# fig.tight_layout()
# fig.savefig(paths.figures / "random_dark_frames.pdf")
