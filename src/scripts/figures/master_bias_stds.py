import paths
import numpy as np
import ccdproc as ccdp
import figure_utils as utils
import matplotlib.pyplot as plt

TRIM = 20
N_FRAMES = 40

means = np.zeros(N_FRAMES)
stds = np.zeros(N_FRAMES)

base_path = paths.data / "processed_photometry" / "calibration" / "bias"

for i in range(N_FRAMES):
    file_path = base_path / f"master_bias_{i+1}.fits"
    master_bias = ccdp.CCDData.read(file_path)
    master_bias_data = master_bias.data.copy()[TRIM:-TRIM, TRIM:-TRIM]
    means[i] = np.mean(master_bias_data)
    stds[i] = np.std(master_bias_data) / np.sqrt(i + 1)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].errorbar(
    np.arange(N_FRAMES) + 1, means, yerr=stds, fmt="o", color=utils.default_colors[0]
)
ax[0].set_xlabel("Number of Biases Frames", fontsize=16)
ax[0].set_ylabel("Mean [ADU]", fontsize=16)
ax[0].set_title("Mean Signal vs. Frames", fontsize=20)

ax[1].scatter(np.arange(N_FRAMES) + 1, stds, color=utils.default_colors[0])
ax[1].set_xlabel("Number of Biases Frames", fontsize=16)
ax[1].set_ylabel("Standard Deviation [ADU]", fontsize=16)
ax[1].set_title("STD vs. Frames", fontsize=20)
fig.tight_layout()
fig.savefig(paths.figures / "master_bias_stds.pdf")
