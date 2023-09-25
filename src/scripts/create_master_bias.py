import paths
import utils
import ccdproc as ccdp
import matplotlib.pyplot as plt

DIM = 1030
raw_calib_path = paths.data / "raw_photometry" / "CALIB"
save_path = paths.data / "processed_photometry" / "CALIB" / "bias"

calib_collection = ccdp.ImageFileCollection(location=raw_calib_path)

bias_filter = {
    "object": "BIAS",
    "naxis1": DIM,
    "naxis2": DIM,
}

bias_collection = calib_collection.filter(**bias_filter)
biases = list(bias_collection.ccds(ccd_kwargs={"unit": "adu"}))

master_bias = utils.create_master(
    biases, image_type="BIAS", save=True, save_path=save_path
)

fig, ax = plt.subplots(figsize=(10, 10))
utils.show_image(master_bias, fig=fig, ax=ax, cbar_label="Signal [ADU]")
ax.set_title("Master Bias", fontsize=20)
ax.set_xlabel("X [pixels]", fontsize=16)
ax.set_ylabel("Y [pixels]", fontsize=16)
fig.tight_layout()
fig.savefig(paths.figures / "master_bias.pdf")
