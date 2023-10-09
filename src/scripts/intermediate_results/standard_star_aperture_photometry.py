import paths
import numpy as np
import pandas as pd
import ccdproc as ccdp
import photutils as phot
import astropy.units as u
import processing_utils as putils

from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

fits_path = paths.data / "processed_photometry" / "science" / "standard_stars" / "fits"
fwhm_path = paths.data / "processed_photometry" / "numbers" / "standard_star_fwhm"
output_path = (
    paths.data / "processed_photometry" / "numbers" / "standard_star_aperture_phot"
)
output_path.mkdir(parents=True, exist_ok=True)

init_centroids = (
    (817, 531),
    (286, 237),
    (313, 667),
)
star_names = ["0", "B", "C"]
flat_type = "sky"
filter_names = ["B", "V", "R"]
fwhm_func = "Gaussian"

mag_and_mag_errs_column_names = []
mag_and_mag_errs = np.zeros((len(star_names), 2 * len(filter_names)))

for filter_name in filter_names:
    # Load image
    image_path = fits_path / (filter_name + "_" + flat_type + "_image.fits")
    image = ccdp.CCDData.read(image_path)
    exposure_time = image.header["EXPTIME"] * u.s

    # Load FWHM
    fwhms = np.zeros((2, len(init_centroids)))
    for j, star_name in enumerate(star_names):
        fwhm_df = pd.read_csv(
            fwhm_path / f"{star_name}_{filter_name}_fwhm.csv", index_col=0
        )
        fwhm = fwhm_df.loc[fwhm_func, "FWHM"]
        fwhm_err = fwhm_df.loc[fwhm_func, "FWHM_err"]
        fwhms[:, j] = fwhm, fwhm_err

    # Weighted mean FWHM
    fwhm = np.average(fwhms[0], weights=1 / fwhms[1] ** 2) * u.pix
    fwhm_err = np.sqrt(1 / np.sum(1 / fwhms[1] ** 2)) * u.pix
    print(f"\n{filter_name} FWHM: {fwhm.value:.2f} +/- {fwhm_err.value:.2f} {u.pix}")

    # Define apertures
    aperture_radius = 3 * fwhm
    aperture_area = np.pi * aperture_radius**2
    apertures = CircularAperture(init_centroids, r=fwhm.value)

    print(f"Aperture area: {aperture_area:.2f}")
    # Define annulus
    annulus_radius_inner = 4 * fwhm
    annulus_radius_outer = 6 * fwhm
    annulus_area = np.pi * (annulus_radius_outer**2 - annulus_radius_inner**2)
    annulus_apertures = CircularAnnulus(
        init_centroids,
        r_in=annulus_radius_inner.value,
        r_out=annulus_radius_outer.value,
    )
    print(f"Annulus area: {annulus_area:.2f}")

    # Create sigclipped mean bkg
    annulus_masks = annulus_apertures.to_mask(method="subpixel", subpixels=5)
    annulus_masks = [
        np.logical_not(mask.to_image(image.shape)) for mask in annulus_masks
    ]
    sig_clipper = SigmaClip(
        sigma=3.0, maxiters=None, cenfunc="median", stdfunc="mad_std"
    )

    bkg_means = np.zeros(len(init_centroids))
    bkg_mean_errs = np.zeros(len(init_centroids))

    for i, mask in enumerate(annulus_masks):
        image_mask = np.logical_or(image.mask, mask)
        masked_image_array = np.ma.array(image.data, mask=image_mask)

        sigclipped_image = sig_clipper(masked_image_array, masked=True)
        sigclipped_uncertainty = np.ma.array(
            image.uncertainty.array, mask=sigclipped_image.mask
        )

        bkg_mean = np.ma.mean(sigclipped_image)
        bkg_mean_err = (
            np.ma.sqrt(np.ma.sum(sigclipped_uncertainty**2))
            / sigclipped_image.count()
        )

        bkg_means[i] = bkg_mean
        bkg_mean_errs[i] = bkg_mean_err

    bkg_means *= u.electron
    bkg_means /= annulus_area
    bkg_mean_errs *= u.electron
    bkg_mean_errs /= annulus_area

    # Unsubtracted aperture photometry
    phot_table = aperture_photometry(
        data=image,
        apertures=apertures,
        # error=image.uncertainty,
        method="subpixel",
        subpixels=5,
    )

    phot_table["annulus_means"] = bkg_means
    phot_table["annulus_mean_errs"] = bkg_mean_errs

    phot_table["aperture_bkg"] = bkg_means * aperture_area
    phot_table["aperture_bkg_err"] = bkg_mean_errs * aperture_area

    phot_table["aperture_sum_bkg_subtracted"] = (
        phot_table["aperture_sum"] - phot_table["aperture_bkg"]
    )
    phot_table["aperture_sum_bkg_subtracted_err"] = np.sqrt(
        phot_table["aperture_sum_err"] ** 2 + phot_table["aperture_bkg_err"] ** 2
    )

    phot_table["exposure_time"] = exposure_time
    phot_table["count_rate"] = phot_table["aperture_sum_bkg_subtracted"] / exposure_time
    phot_table["count_rate_err"] = (
        phot_table["aperture_sum_bkg_subtracted_err"] / exposure_time
    )

    estimated_magnitudes = -2.5 * np.log10(phot_table["count_rate"].value) + 25
    estimated_magnitudes_err = (
        2.5 / np.log(10) * phot_table["count_rate_err"] / phot_table["count_rate"]
    )

    mag_and_mag_errs[:, 2 * filter_names.index(filter_name)] = estimated_magnitudes
    mag_and_mag_errs[
        :, 2 * filter_names.index(filter_name) + 1
    ] = estimated_magnitudes_err
    mag_and_mag_errs_column_names += [
        filter_name + "_mag",
        filter_name + "_mag_err",
    ]

    phot_table["estimated_mag"] = estimated_magnitudes * u.mag
    phot_table["estimated_mag_err"] = estimated_magnitudes_err * u.mag

    phot_table["id"] = star_names

    df = phot_table.to_pandas(index="id")

    df.to_csv(
        output_path / f"{filter_name}_{flat_type}_aperture_phot.csv",
        index=True,
    )

mag_and_mag_errs_df = pd.DataFrame(
    mag_and_mag_errs, index=star_names, columns=mag_and_mag_errs_column_names
)

mag_and_mag_errs_df.to_csv(
    output_path / f"{flat_type}_aperture_phot.csv",
    index=True,
)
