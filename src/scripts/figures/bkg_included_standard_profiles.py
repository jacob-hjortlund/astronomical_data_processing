import paths
import numpy as np
import colorcet as cc
import ccdproc as ccdp
import scipy.stats as stats
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
    (817, 531),
    (286, 237),
    (313, 667),
    # (285, 236),  # Standard Star 1
    # (331, 335),  # Standard Star 2
    # (817, 530),  # Standard Star 3
)
star_names = ["0", "B", "C"]

flat_type = "sky"
filter_names = ["B", "V", "R"]
colormaps = [cc.cm.kb, cc.cm.kg, cc.cm.kr]

for filter_name, colormap in zip(filter_names, colormaps):
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

        xc, yc = cen.centroid_2dg(data, uncertainty)

        show_colorbar = False
        if i == 2:
            show_colorbar = True

        utils.show_image(
            image,
            ax=ax[0, i],
            fig=fig,
            cmap=colormap,
            show_colorbar=show_colorbar,
        )
        print(f"Star {i+1} {filter_name}-centroid: ({xc:.2f}, {yc:.2f})")
        ax[0, i].plot(
            x0, y0, "x", color="gray", ms=10, mew=2, zorder=1000, label="Init. Centroid"
        )
        ax[0, i].plot(
            xc + xmin,
            yc + ymin,
            "+",
            color="white",
            ms=10,
            mew=2,
            zorder=1000,
            label="2D Gauss Centroid",
        )
        ax[0, i].legend(loc="upper right", fontsize=12)

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
        mean = gauss_fit.parameters
        mvg = stats.multivariate_normal(mean, cov)
        gauss_samples = mvg.rvs(1000)
        gauss_counts = np.zeros((1000, len(MOD_RADII)))
        for j, sample in enumerate(gauss_samples):
            gauss_counts[j] = mod.models.Gaussian1D(*sample)(MOD_RADII)

        gauss_median = np.median(gauss_counts, axis=0)
        gauss_upper = np.percentile(gauss_counts, 84, axis=0)
        gauss_lower = np.percentile(gauss_counts, 16, axis=0)

        std = gauss_fit.stddev.value
        gauss_fwhm = 2 * np.sqrt(2 * np.log(2)) * std
        gauss_fwhm_unc = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(np.diag(cov))[2]

        gauss_fwhm_string = utils.convert_variable_to_latex(
            gauss_fwhm, gauss_fwhm_unc, sigfigs=1
        )

        # Fit 1D Moffat
        moffat_init = mod.models.Moffat1D(
            amplitude=np.nanmax(counts),
            x_0=0,
            gamma=1,
            alpha=1,
        )
        moffat_fit = fitter(moffat_init, mirrored_radii, mirrored_counts)

        cov = fitter.fit_info["param_cov"]
        mean = moffat_fit.parameters
        mvg = stats.multivariate_normal(mean, cov)
        moffat_samples = mvg.rvs(1000)
        moffat_counts = np.zeros((1000, len(MOD_RADII)))
        for j, sample in enumerate(moffat_samples):
            moffat_counts[j] = mod.models.Moffat1D(*sample)(MOD_RADII)

        moffat_median = np.median(moffat_counts, axis=0)
        moffat_upper = np.percentile(moffat_counts, 84, axis=0)
        moffat_lower = np.percentile(moffat_counts, 16, axis=0)

        alpha = moffat_fit.alpha.value
        gamma = moffat_fit.gamma.value

        moffat_fwhm = 2 * alpha * np.sqrt(2 ** (1 / gamma) - 1)
        moffat_alpha_variance_term = (
            np.diag(cov)[3] * (2 * np.sqrt(2 ** (1 / gamma) - 1)) ** 2
        )
        moffat_gamma_variance_term = (
            np.diag(cov)[2]
            * (
                (alpha * np.log(2) * 2 ** (1 / gamma))
                / (np.sqrt(2 ** (1 / gamma) - 1) * gamma**2)
            )
            ** 2
        )
        moffat_fwhm_unc = np.sqrt(
            moffat_alpha_variance_term + moffat_gamma_variance_term
        )

        moffat_fwhm_string = utils.convert_variable_to_latex(
            moffat_fwhm, moffat_fwhm_unc, sigfigs=1
        )

        ax[1, i].errorbar(
            profile_radii,
            counts,
            yerr=uncertainty,
            fmt="o",
            color="black",
        )

        ax[1, i].plot(
            MOD_RADII,
            gauss_fit(MOD_RADII),
            color=utils.default_colors[0],
            label=r"Gauss, FWHM = " + gauss_fwhm_string,
            lw=2,
        )
        ax[1, i].fill_between(
            MOD_RADII,
            gauss_lower,
            gauss_upper,
            color=utils.default_colors[0],
            alpha=0.3,
        )
        ax[1, i].plot(
            MOD_RADII,
            moffat_fit(MOD_RADII),
            color=utils.default_colors[1],
            label=r"Moffat, FWHM = " + moffat_fwhm_string,
            lw=2,
        )
        ax[1, i].fill_between(
            MOD_RADII,
            moffat_lower,
            moffat_upper,
            color=utils.default_colors[1],
            alpha=0.3,
        )

        ax[0, i].set_title(f"PG 1323-086-{star_names[i]}", fontsize=14)
        ax[0, i].set_xlim(xmin, xmax)
        ax[0, i].set_ylim(ymin, ymax)
        ax[0, i].set_xlabel("X [pixels]", fontsize=14)
        if i == 0:
            ax[0, i].set_ylabel("Y [pixels]", fontsize=14)
            ax[1, i].set_ylabel("Electron Counts", fontsize=14)

        ax[1, i].legend(loc="upper right", fontsize=10)

        ax[1, i].set_xlabel("Radius [Pixels]", fontsize=14)

    fig.tight_layout()
    fig.savefig(
        paths.figures / f"bkg_included_standard_{filter_name}_profiles.pdf",
        bbox_inches="tight",
    )
