import paths
import ccdproc as ccdp
import figure_utils as utils
import matplotlib.pyplot as plt

base_path = paths.data / "processed_photometry" / "science" / "observations"

filters = ["V", "R"]
flat_type = "lamp"

fig, ax = plt.subplots(
    ncols=2, nrows=2, figsize=(6 * 3, 6 * 3), sharex=True, sharey=True
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
ax[0, 0].set_ylabel("V", fontsize=24)
ax[1, 0].set_ylabel("R", fontsize=24)
ax[1, 0].set_xlabel("X [pixels]", fontsize=24)
ax[1, 1].set_xlabel("X [pixels]", fontsize=24)
fig.suptitle("Lamp Flat Reduced Science Images", fontsize=24)
fig.tight_layout()
fig.savefig(
    paths.figures / "lampflat_calibrated_science_images.pdf",
    bbox_inches="tight",
)
