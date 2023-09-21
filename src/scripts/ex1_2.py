import paths
import utils
import numpy as np
import ccdproc as ccdp
import matplotlib.pyplot as plt

DIM = 1030
rng = np.random.default_rng(42)
raw_calib_path = paths.data / "raw_photometry" / "CALIB"

calib_collection = ccdp.ImageFileCollection(
    location=raw_calib_path,
)

bias_filter = {
    "object": "BIAS",
    "naxis1": DIM,
    "naxis2": DIM,
}

bias_collection = calib_collection.filter(**bias_filter)
biases = list(bias_collection.ccds(ccd_kwargs={"unit": "adu"}))
random_bias_idx = rng.choice(len(biases), size=4, replace=False)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

for i, idx in enumerate(random_bias_idx):
    utils.show_image(
        biases[idx],
        fig=fig,
        ax=ax[i // 2, i % 2],
    )
    ax[i // 2, i % 2].set_title(f"Bias {i + 1}", fontsize=16)

fig.suptitle("Randomly Selected Bias Frames", fontsize=20)
fig.tight_layout()

plt.savefig(paths.figures / "random_bias_frames.pdf")
