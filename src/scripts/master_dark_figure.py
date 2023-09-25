import os
import paths
import utils
import numpy as np
import ccdproc as ccdp
import matplotlib.pyplot as plt

path = paths.data / "processed_photometry" / "CALIB" / "darks" / "master_DARK.fits"
master_dark = ccdp.CCDData.read(path)

fig, ax = plt.subplots(figsize=(10, 10))
utils.show_image(master_dark, fig=fig, ax=ax, cbar_label="Signal [ADU]")
ax.set_title("Master Dark", fontsize=20)
ax.set_xlabel("X [pixels]", fontsize=16)
ax.set_ylabel("Y [pixels]", fontsize=16)
fig.tight_layout()
fig.savefig(paths.figures / "master_dark.pdf")
