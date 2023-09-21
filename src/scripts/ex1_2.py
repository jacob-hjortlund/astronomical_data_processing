import paths
import utils
import numpy as np
import ccdproc as ccdp
import matplotlib.pyplot as plt

DIM = 1030
rng = np.random.default_rng(42)
raw_calib_path = paths.data / "raw_photometry" / "CALIB"
files = [
    "EFOSC.2000-12-29T22:09:43.891.fits",
    "EFOSC.2000-12-30T05:17:49.057.fits",
    "EFOSC.2000-12-28T22:09:41.589.fits",
    "EFOSC.2000-12-29T05:18:31.412.fits",
]

calib_collection = ccdp.ImageFileCollection(location=raw_calib_path, filenames=files)

bias_filter = {
    "object": "BIAS",
    "naxis1": DIM,
    "naxis2": DIM,
}

bias_collection = calib_collection.filter(**bias_filter)
biases = list(bias_collection.ccds(ccd_kwargs={"unit": "adu"}))

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()
for i, bias in enumerate(biases):
    print(f"Mean bias {i + 1}: {np.mean(bias.data)}")
    utils.show_image(bias, fig=fig, ax=ax[i])
    ax[i].set_title(f"Bias {i + 1}", fontsize=16)

fig.suptitle("Randomly Selected Bias Frames", fontsize=20)
fig.tight_layout()

plt.savefig(paths.figures / "random_bias_frames.pdf")
