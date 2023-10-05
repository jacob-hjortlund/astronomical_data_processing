import paths
import numpy as np
import ccdproc as ccdp
import figure_utils as utils
import matplotlib.pyplot as plt

DIM = 1030
raw_bias_path = paths.data / "raw_photometry" / "CALIB" / "BIAS"
files = [
    "EFOSC.2000-12-30T05:17:49.057.fits",
    "EFOSC.2000-12-29T04:39:50.783.fits",
    "EFOSC.2000-12-28T22:11:19.687.fits",
    "EFOSC.2000-12-30T04:19:55.000.fits",
]

bias_collection = ccdp.ImageFileCollection(location=raw_bias_path, filenames=files)

biases = list(bias_collection.ccds(ccd_kwargs={"unit": "adu"}))

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()
for i, bias in enumerate(biases):
    print(f"Mean bias {i + 1}: {np.mean(bias.data)}")
    utils.show_image(bias, fig=fig, ax=ax[i], cbar_label="Signal [ADU]")
    ax[i].set_title(f"Bias {i + 1}", fontsize=16)

fig.suptitle("Randomly Selected Bias Frames", fontsize=20)
fig.tight_layout()
fig.savefig(paths.figures / "random_bias_frames.pdf")
