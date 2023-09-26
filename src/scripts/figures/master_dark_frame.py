import paths
import ccdproc as ccdp
import figure_utils as utils
import matplotlib.pyplot as plt

path = (
    paths.data / "processed_photometry" / "calibration" / "darks" / "master_dark.fits"
)
master_dark = ccdp.CCDData.read(path)

fig, ax = plt.subplots(figsize=(10, 10))
utils.show_image(master_dark, fig=fig, ax=ax, cbar_label="Signal [ADU]")
ax.set_title("Master Dark", fontsize=20)
ax.set_xlabel("X [pixels]", fontsize=16)
ax.set_ylabel("Y [pixels]", fontsize=16)
fig.tight_layout()
fig.savefig(paths.figures / "master_dark_frame.pdf")
