import os
import paths
import utils
import ccdproc as ccdp
import matplotlib.pyplot as plt

filters = ["B", "V", "R"]
master_skyflat_path = paths.data / "processed_photometry" / "CALIB" / "flats" / "sky"

fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

for i, filter_name in enumerate(filters):
    filter_path = master_skyflat_path / filter_name
    master_skyflat = ccdp.CCDData.read(filter_path / "master_SKYFLAT.fits")
    master_skyflat = ccdp.trim_image(
        master_skyflat[20:-20, 20:-20],
    )
    utils.show_image(
        master_skyflat,
        fig=fig,
        ax=ax[i],
        cbar_label="Signal",
    )
    ax[i].set_title(f"Filter ({filter_name})", fontsize=20)
    ax[i].set_xlabel("X [pixels]", fontsize=16)
    ax[i].set_ylabel("Y [pixels]", fontsize=16)

fig.suptitle("Master Sky Flats", fontsize=24)
fig.tight_layout()
fig.savefig(paths.figures / "master_skyflat.pdf")
