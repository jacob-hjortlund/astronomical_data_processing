import paths
import utils
import numpy as np
import ccdproc as ccdp
import matplotlib.pyplot as plt

DIM = 1030
TRIM = 12
raw_calib_path = paths.data / "raw_photometry" / "CALIB"
save_path = paths.data / "processed_photometry" / "CALIB" / "bias"

calib_collection = ccdp.ImageFileCollection(location=raw_calib_path)

bias_filter = {
    "object": "BIAS",
    "naxis1": DIM,
    "naxis2": DIM,
}

bias_collection = calib_collection.filter(**bias_filter)
biases = list(bias_collection.ccds(ccd_kwargs={"unit": "adu"}))

n_frames = len(biases)
master_biases = []
means = np.zeros(n_frames)
stds = np.zeros(n_frames)
for i in range(n_frames):
    master_bias = utils.create_master(
        biases[: i + 1], image_type="BIAS", save=True, save_path=save_path
    )
    master_bias_data = master_bias.data[TRIM:-TRIM, TRIM:-TRIM]
    means[i] = np.mean(master_bias_data)
    stds[i] = np.std(master_bias_data) / np.sqrt(i + 1)
    master_biases.append(master_bias)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].errorbar(
    np.arange(n_frames) + 1, means, yerr=stds, fmt="o", color=utils.default_colors[0]
)
ax[0].set_xlabel("Number of Biases Frames", fontsize=16)
ax[0].set_ylabel("Mean [ADU]", fontsize=16)
ax[0].set_title("Mean Signal vs. Number of Bias Frames", fontsize=20)

ax[1].scatter(np.arange(n_frames) + 1, stds, color=utils.default_colors[0])
ax[1].set_xlabel("Number of Biases Frames", fontsize=16)
ax[1].set_ylabel("Standard Deviation [ADU]", fontsize=16)
ax[1].set_title("Standard Deviation vs. Number of Bias Frames", fontsize=20)
fig.tight_layout()
fig.savefig(paths.figures / "bias_frame_stats.pdf")

fig, ax = plt.subplots(figsize=(10, 10))
utils.show_image(master_bias, fig=fig, ax=ax, cbar_label="Signal [ADU]")
ax.set_title("Master Bias", fontsize=20)
ax.set_xlabel("X [pixels]", fontsize=16)
ax.set_ylabel("Y [pixels]", fontsize=16)
fig.tight_layout()
fig.savefig(paths.figures / "master_bias.pdf")
