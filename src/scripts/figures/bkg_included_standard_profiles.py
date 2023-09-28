import paths
import numpy as np
import colorcet as cc
import ccdproc as ccdp
import figure_utils as utils
import astropy.modeling as mod
import matplotlib.pyplot as plt
import photutils.centroids as cen
import photutils.profiles as prof

DELTA = 25
MAX_RADII = 15
EDGE_RADII = np.arange(MAX_RADII)
MOD_RADII = np.linspace(0, MAX_RADII, 1000)

base_path = paths.data / "processed_photometry" / "science" / "standard_stars"

init_centroids = (
    (285, 236),  # Standard Star 1
    (331, 335),  # Standard Star 2
    (817, 530),  # Standard Star 3
)

flat_type = "sky"
filter_name = "B"  # ["B", "V", "R"]
colormap = cc.cm.kb

image_path = base_path / (filter_name + "_" + flat_type.lower() + "_image.fits")
image = ccdp.CCDData.read(image_path)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

for i, init_centroid in enumerate(init_centroids):
    x0, y0 = init_centroid
    xmax, xmin = x0 + DELTA, x0 - DELTA
    ymax, ymin = y0 + DELTA, y0 - DELTA

    trimmed_image = ccdp.trim_image(
        image[ymin:ymax, xmin:xmax],
    )
    data = trimmed_image.data
    uncertainty = trimmed_image.uncertainty.array

    xc, yc = cen.centroid_1dg(data, uncertainty)

    show_colorbar = False
    if i == 2:
        show_colorbar = True

    ax[0, i].set_title(f"Star {i+1}", fontsize=14)
    utils.show_image(
        image,
        ax=ax[0, i],
        fig=fig,
        cmap=colormap,
        show_colorbar=show_colorbar,
    )
    ax[0, i].set_xlim(xmin, xmax)
    ax[0, i].set_ylim(ymin, ymax)
    ax[0, i].set_xlabel("X [pixels]", fontsize=14)

    profile = prof.RadialProfile(
        data, (xc, yc), EDGE_RADII, error=uncertainty, mask=None, method="subpixel"
    )

    counts = profile.profile
    mirrored_counts = np.concatenate((counts[::-1], counts))

    uncertainty = profile.profile_error
    mirrored_uncertainty = np.concatenate((uncertainty[::-1], uncertainty))

    profile_radii = profile.radius
    mirrored_radii = np.concatenate((-profile_radii[::-1], profile_radii))

    fitter = mod.fitting.LevMarLSQFitter()
    # Fit 1D Gauss
    gauss_init = mod.models.Gaussian1D(
        amplitude=np.nanmax(counts),
        mean=0,
        stddev=1,
    )
    gauss_fit = fitter(gauss_init, mirrored_radii, mirrored_counts)

    # Fit 1D Moffat
    moffat_init = mod.models.Moffat1D(
        amplitude=np.nanmax(counts),
        x_0=0,
        gamma=1,
        alpha=1,
    )
    moffat_fit = fitter(moffat_init, mirrored_radii, mirrored_counts)

    ax[1, i].errorbar(
        profile_radii,
        counts,
        yerr=uncertainty,
        fmt="o",
        color="black",
    )
    print("\nGauss")
    print(gauss_fit(EDGE_RADII))
    print("\nMoffat")
    print(moffat_fit(EDGE_RADII))

    ax[1, i].plot(
        MOD_RADII,
        gauss_fit(MOD_RADII),
        color=utils.default_colors[0],
        label="Gaussian",
        lw=2,
    )
    ax[1, i].plot(
        MOD_RADII,
        moffat_fit(MOD_RADII),
        color=utils.default_colors[1],
        label="Moffat",
        lw=2,
    )
    if i == 0:
        ax[0, i].set_ylabel("Y [pixels]", fontsize=14)
        ax[1, i].set_ylabel("Electron Counts", fontsize=14)
        ax[1, i].legend(loc="upper right", fontsize=14)

    ax[1, i].set_xlabel("Radius [Pixels]", fontsize=14)

fig.tight_layout()
fig.savefig(
    paths.figures / "bkg_included_standard_profiles.pdf",
    bbox_inches="tight",
)
