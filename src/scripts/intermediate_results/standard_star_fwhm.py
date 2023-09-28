import paths
import numpy as np
import pandas as pd
import colorcet as cc
import ccdproc as ccdp
import astropy.modeling as mod
import processing_utils as utils
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

output_path = paths.data / "processed_photometry" / "numbers" / "standard_stars_fwhm"

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
    std = gauss_fit.stddev.value
    gauss_fwhm = 2 * np.sqrt(2 * np.log(2)) * std

    # Fit 1D Moffat
    moffat_init = mod.models.Moffat1D(
        amplitude=np.nanmax(counts),
        x_0=0,
        gamma=1,
        alpha=1,
    )
    moffat_fit = fitter(moffat_init, mirrored_radii, mirrored_counts)
    alpha = moffat_fit.alpha.value
    gamma = moffat_fit.gamma.value
    moffat_fwhm = 2 * alpha * np.sqrt(2 ** (1 / gamma) - 1)

    values_arr = np.array(
        [
            [std, np.nan, gauss_fwhm],
            [alpha, gamma, moffat_fwhm],
        ]
    )
    col_names = ["par_1", "par_2", "FWHM"]
    row_names = ["Gaussian", "Moffat"]

    utils.save_array_to_csv(
        array=values_arr,
        column_names=col_names,
        index_names=row_names,
        filename=f"standard_star_{i+1}_fwhm.csv",
        path=output_path,
        overwrite=True,
    )
