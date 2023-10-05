import paths
import numpy as np
import ccdproc as ccdp
import scipy.stats as stats
import figure_utils as utils
import matplotlib.pyplot as plt

DIM = 1030
raw_bias_path = paths.data / "raw_photometry" / "CALIB" / "BIAS"
bias_collection = ccdp.ImageFileCollection(location=raw_bias_path)

biases = list(bias_collection.ccds(ccd_kwargs={"unit": "adu"}))

frame_1 = biases[0]
frame_2 = biases[1]
frame_diff = frame_1.subtract(frame_2)
frames = [frame_1, frame_2, frame_diff]
frame_names = ["Frame 1", "Frame 2", "Frame 1 - Frame 2"]

fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
for i, (frame, frame_name) in enumerate(zip(frames, frame_names)):
    frame_mean = frame.data.mean()
    frame_std = frame.data.std()
    if i == 2:
        frame_std /= np.sqrt(2)
    xlim = (frame_mean - 5 * frame_std, frame_mean + 5 * frame_std)

    idx_clipped = (frame.data > xlim[0]) & (frame.data < xlim[1])
    frame_data_clipped = frame.data[idx_clipped]
    frame_mean = frame_data_clipped.mean()
    frame_std = frame_data_clipped.std()
    # if i == 2:
    # frame_std /= np.sqrt(2)

    x_range = np.linspace(*xlim, 1000)
    y = stats.norm.pdf(x_range, loc=frame_mean, scale=frame_std)

    ax[i].hist(
        frame_data_clipped.flatten(),
        bins=40,
        density=True,
        color=utils.default_colors[i],
        range=xlim,
    )
    ax[i].plot(x_range, y, color="k", linewidth=3)
    ax[i].set_xlim(*xlim)
    ax[i].set_title(
        f"{frame_name}\nMean: {frame_mean:.2f} ADU, STD: {frame_std:.2f} ADU",
        fontsize=16,
    )
    ax[i].set_xlabel("Signal [ADU]", fontsize=16)
    ax[i].set_ylabel("Density", fontsize=16)

fig.tight_layout()
fig.savefig(paths.figures / "bias_frame_histograms.pdf")
