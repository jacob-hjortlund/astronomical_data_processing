import paths
import utils
import ccdproc as ccdp
import matplotlib.pyplot as plt

master_bias = ccdp.CCDData.read(
    paths.data / "processed_photometry" / "CALIB" / "bias" / "master_BIAS_40.fits",
    unit="adu",
)

fig, ax = plt.subplots(figsize=(10, 10))
utils.show_image(master_bias, fig=fig, ax=ax, cbar_label="Signal [ADU]")
ax.set_title("Master Bias", fontsize=20)
ax.set_xlabel("X [pixels]", fontsize=16)
ax.set_ylabel("Y [pixels]", fontsize=16)
fig.tight_layout()
fig.savefig(paths.figures / "master_bias.pdf")
