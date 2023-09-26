import paths
import ccdproc as ccdp
import figure_utils as utils
import matplotlib.pyplot as plt

base_path = paths.data / "processed_photometry" / "science" / "standard_stars"

filters = ["B", "V", "R"]
flat_type = "sky"

fig, ax = plt.subplots(
    ncols=2, nrows=3, figsize=(6 * 3, 9 * 3), sharex=True, sharey=True
)
for i, filter_name in enumerate(filters):
    image_path = base_path / (filter_name + "_" + flat_type.lower() + "_image.fits")
    image = ccdp.CCDData.read(image_path)
    uncertainty = ccdp.CCDData(image.uncertainty.array, unit="adu")

    utils.show_image(
        image,
        ax=ax[i, 0],
        fig=fig,
        cbar_label=r"Signal [e$^-$]",
    )
    utils.show_image(
        uncertainty,
        ax=ax[i, 1],
        fig=fig,
        cbar_label=r"Uncertainty [e$^-$]",
    )

ax[0, 0].set_title("Image", fontsize=24)
ax[0, 1].set_title("Uncertainty", fontsize=24)
ax[0, 0].set_ylabel("B", fontsize=24)
ax[1, 0].set_ylabel("V", fontsize=24)
ax[2, 0].set_ylabel("R", fontsize=24)
ax[2, 0].set_xlabel("X [pixels]", fontsize=24)
ax[2, 1].set_xlabel("X [pixels]", fontsize=24)
fig.suptitle("Sky Flat Reduced Standard Star Images", fontsize=24)
fig.tight_layout()
fig.savefig(
    paths.figures / "skyflat_calibrated_standard_star_images.pdf",
    bbox_inches="tight",
)
