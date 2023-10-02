import paths
import numpy as np
import pandas as pd
import number_utils as utils

result_path = paths.data / "processed_photometry" / "numbers" / "standard_stars_fwhm"
filters = ["B", "V", "R"]
n_stars = 3

# TODO: CONVERT FROM PIXELS TO ARCSECONDS

for filter_name in filters:
    fwhms = np.zeros((2, n_stars))
    fwhm_errs = np.zeros((2, n_stars))

    for i in range(n_stars):
        file_name = f"standard_star_{filter_name}_{i+1}_fwhm.csv"
        file_path = result_path / file_name
        result = pd.read_csv(file_path, index_col=0)

        fwhms[0, i] = result.loc["Gaussian", "FWHM"]
        fwhms[1, i] = result.loc["Moffat", "FWHM"]
        fwhm_errs[0, i] = result.loc["Gaussian", "FWHM_err"]
        fwhm_errs[1, i] = result.loc["Moffat", "FWHM_err"]

    ## Weighted mean
    fwhm_mean, fwhm_err = utils.weighted_average(fwhms, fwhm_errs, axis=1)

    print(f"\nFilter: {filter_name}")
    print(f"FWHM Gaussian: {fwhm_mean[0]} +/- {fwhm_err[0]}")
    print(f"FWHM Moffat: {fwhm_mean[1]} +/- {fwhm_err[1]}")

    for i, model in enumerate(["Gaussian", "Moffat"]):
        utils.save_variable_to_latex(
            variable=fwhm_mean[i],
            variable_error=fwhm_err[i],
            variable_name=f"FWHM_{model}",
            filename=f"fwhm_{filter_name}.dat",
            path=paths.output,
            sigfigs=1,
        )
