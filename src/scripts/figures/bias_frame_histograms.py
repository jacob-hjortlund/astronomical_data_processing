import paths
import numpy as np
import ccdproc as ccdp
import scipy.stats as stats
import figure_utils as utils
import matplotlib.pyplot as plt

DIM = 1030
raw_calib_path = paths.data / "raw_photometry" / "CALIB"
calib_collection = ccdp.ImageFileCollection(location=raw_calib_path)

bias_filter = {
    "object": "BIAS",
    "naxis1": DIM,
    "naxis2": DIM,
}

bias_collection = calib_collection.filter(**bias_filter)
biases = list(bias_collection.ccds(ccd_kwargs={"unit": "adu"}))

frame_1 = biases[0]
frame_2 = biases[2]
frame_diff = frame_1.subtract(frame_2)
frames = [frame_1, frame_2, frame_diff]
frame_names = ["Frame 1", "Frame 2", "Frame 1 - Frame 2"]

fig, ax = plt.subplots(2,3, figsize=(15, 10))
for i, (frame, frame_name) in enumerate(zip(frames, frame_names)):
    frame_mean = frame.data.mean()
    frame_std = frame.data.std()
    xlim = (frame_mean - 5 * frame_std, frame_mean + 5 * frame_std)

    idx_clipped = (frame.data > xlim[0]) & (frame.data < xlim[1])
    frame_data_clipped = frame.data[idx_clipped]
    frame_mean = frame_data_clipped.mean()
    frame_std = frame_data_clipped.std()

    x_range = np.linspace(*xlim, 1000)
    y = stats.norm.pdf(x_range, loc=frame_mean, scale=frame_std)

#FIGURE SHOWING THE BIAS FRAMES HISTOGRAMS
    ax[1,i].hist(
        frame_data_clipped.flatten(),
        bins=40,
        density=True,
        color=utils.default_colors[i],
        range=xlim,
    )
    ax[1,i].plot(x_range, y, color="k", linewidth=3)
    ax[1,i].set_xlim(*xlim)
    ax[1,i].set_title(
        f"{frame_name}\nMean: {frame_mean:.2f} ADU, STD: {frame_std:.2f} ADU",
        fontsize=16)
    ax[1,i].set_xlabel("Signal [ADU]", fontsize=16)
    ax[1,i].set_ylabel("Density", fontsize=16)

#FIGURE SHOWING THE BIAS FRAMES DATA
    utils.show_image(
        frame,
        ax=ax[0,i],
        fig=fig
    )
    #ax[0,i].imshow(frame.data, norm='log', vmin=140, vmax=160)
    ax[0,i].set_xlabel("Pixel", fontsize=10)
    ax[0,i].set_ylabel("Pixel", fontsize=10)
    ax[0,i].set_title(
        f"{frame_name}",fontsize=16)
fig.tight_layout()
fig.savefig(paths.figures / "bias_frame_histograms.pdf")