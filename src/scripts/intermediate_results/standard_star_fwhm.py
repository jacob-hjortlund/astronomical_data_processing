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

base_path = paths.data / "processed_photometry" / "science" / "standard_stars" / "fits"

init_centroids = (
    (817, 531),
    (286, 237),
    (313, 667),
)
star_names = ["0", "B", "C"]
flat_type = "sky"
filter_names = ["B", "V", "R"]

output_path = paths.data / "processed_photometry" / "numbers" / "standard_star_fwhm"

for filter_name in filter_names:
    image_path = base_path / (filter_name + "_" + flat_type.lower() + "_image.fits")
    image = ccdp.CCDData.read(image_path)

    for i, init_centroid in enumerate(init_centroids):
        x0, y0 = init_centroid
        xmax, xmin = x0 + DELTA, x0 - DELTA
        ymax, ymin = y0 + DELTA, y0 - DELTA

        trimmed_image = ccdp.trim_image(
            image[ymin:ymax, xmin:xmax],
        )
        data = trimmed_image.data
        uncertainty = trimmed_image.uncertainty.array

        xc, yc = cen.centroid_2dg(data, uncertainty)

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
        cov = fitter.fit_info["param_cov"]

        std = gauss_fit.stddev.value
        std_unc = np.sqrt(np.diag(cov))[2]
        gauss_fwhm = 2 * np.sqrt(2 * np.log(2)) * std
        gauss_fwhm_unc = 2 * np.sqrt(2 * np.log(2)) * std_unc

        # Fit 1D Moffat
        moffat_init = mod.models.Moffat1D(
            amplitude=np.nanmax(counts),
            x_0=0,
            gamma=1,
            alpha=1,
        )
        moffat_fit = fitter(moffat_init, mirrored_radii, mirrored_counts)
        cov = fitter.fit_info["param_cov"]
        alpha = moffat_fit.alpha.value
        gamma = moffat_fit.gamma.value
        gamma_unc = np.sqrt(np.diag(cov))[2]
        alpha_unc = np.sqrt(np.diag(cov))[3]
        moffat_fwhm = 2 * alpha * np.sqrt(2 ** (1 / gamma) - 1)
        moffat_alpha_variance_term = (
            alpha_unc**2 * (2 * np.sqrt(2 ** (1 / gamma) - 1)) ** 2
        )
        moffat_gamma_variance_term = (
            gamma_unc**2
            * (
                (alpha * np.log(2) * 2 ** (1 / gamma))
                / (np.sqrt(2 ** (1 / gamma) - 1) * gamma**2)
            )
            ** 2
        )
        moffat_fwhm_unc = np.sqrt(
            moffat_alpha_variance_term + moffat_gamma_variance_term
        )

        values_arr = np.array(
            [
                [
                    std,
                    std_unc,
                    np.nan,
                    np.nan,
                    gauss_fwhm,
                    gauss_fwhm_unc,
                ],
                [alpha, alpha_unc, gamma, gamma_unc, moffat_fwhm, moffat_fwhm_unc],
            ]
        )
        col_names = ["par_1", "par_1_err", "par_2", "par_2_err", "FWHM", "FWHM_err"]
        row_names = ["Gaussian", "Moffat"]

        utils.save_array_to_csv(
            array=values_arr,
            column_names=col_names,
            index_names=row_names,
            filename=f"{star_names[i]}_{filter_name}_fwhm.csv",
            path=output_path,
            overwrite=True,
        )
