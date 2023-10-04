import paths
import numpy as np
import pandas as pd
import number_utils as utils

result_path = paths.data / "processed_photometry" / "numbers" / "standard_stars_fwhm"
star_names = ["0", "B", "C"]
filters = ["B", "V", "R"]
n_stars = 3
resolution = 0.31  # arcseconds per pixel

# TODO: CONVERT FROM PIXELS TO ARCSECONDS

for filter_name in filters:
    fwhms = np.zeros((2, n_stars))
    fwhm_errs = np.zeros((2, n_stars))

    for i in range(n_stars):
        file_name = f"PG_323_086_{star_names[i]}_{filter_name}_fwhm.csv"
        file_path = result_path / file_name
        result = pd.read_csv(file_path, index_col=0)

        fwhms[0, i] = result.loc["Gaussian", "FWHM"]
        fwhms[1, i] = result.loc["Moffat", "FWHM"]
        fwhm_errs[0, i] = result.loc["Gaussian", "FWHM_err"]
        fwhm_errs[1, i] = result.loc["Moffat", "FWHM_err"]

    ## Weighted mean
    fwhm_mean, fwhm_err = utils.weighted_average(fwhms, fwhm_errs, axis=1)
    seeing_mean = fwhm_mean * resolution
    seeing_err = fwhm_err * resolution

    print(f"\nFilter: {filter_name}")
    print(
        f"FWHM / Seeing Gaussian: {fwhm_mean[0]:.2f} +/- {fwhm_err[0]:.2f} Pixels, {seeing_mean[0]:.2f} +/- {seeing_err[0]:.2f} arcsec"
    )
    print(
        f"FWHM / Seeing Moffat: {fwhm_mean[1]:.2f} +/- {fwhm_err[1]:.2f} Pixels, {seeing_mean[1]:.2f} +/- {seeing_err[1]:.2f} arcsec"
    )

    for i, model in enumerate(["Gaussian", "Moffat"]):
        utils.save_variable_to_latex(
            variable=fwhm_mean[i],
            variable_error=fwhm_err[i],
            variable_name=f"FWHM_{model}",
            filename=f"fwhm_{filter_name}.dat",
            path=paths.output,
            sigfigs=1,
        )
