import paths
import ccdproc as ccdp
import figure_utils as utils
import matplotlib.pyplot as plt

filters = ["V", "R"]
master_skyflat_path = (
    paths.data / "processed_photometry" / "calibration" / "flats" / "lamp"
)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

for i, filter_name in enumerate(filters):
    filter_path = master_skyflat_path / filter_name
    master_skyflat = ccdp.CCDData.read(filter_path / "master_lampflat.fits")
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

fig.suptitle("Master Lamp Flats", fontsize=24)
fig.tight_layout()
fig.savefig(paths.figures / "master_lampflat_frames.pdf")
