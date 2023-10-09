import paths
import emcee as em
import numpy as np
import pandas as pd
import ccdproc as ccdp

from scipy.optimize import minimize


fits_path = paths.data / "processed_photometry" / "science" / "standard_stars" / "fits"
photometry_path = (
    paths.data / "processed_photometry" / "numbers" / "standard_star_aperture_phot"
)
save_path = (
    paths.data / "processed_photometry" / "numbers" / "standard_star_calibration"
)
save_path.mkdir(parents=True, exist_ok=True)

star_names = ["0", "B", "C"]
flat_type = "sky"
filter_names = ["B", "V", "R"]
airmass_correction = np.array([-0.25, -0.13, -0.09])

v_mags = np.array([[13.481, 13.406, 14.003]])
v_mag_errs = np.array([0.0019, 0.0019, 0.0031])

# B-V, V-R
colors = np.array([[-0.140, -0.048], [0.761, 0.426], [0.707, 0.395]])
color_errs = np.array(
    [
        [0.0022, 0.0018],
        [0.0029, 0.0023],
        [0.0027, 0.0024],
    ]
)

# B, V, R along rows, 0, B, C along columns
magnitudes = np.array(
    [
        colors[:, 0] + v_mags[0],
        v_mags[0],
        -colors[:, 1] + v_mags[0],
    ]
)
magnitude_errs = np.array(
    [
        np.sqrt(color_errs[:, 0] ** 2 + v_mag_errs**2),
        v_mag_errs,
        np.sqrt(color_errs[:, 1] ** 2 + v_mag_errs**2),
    ]
)

airmasses = np.zeros(len(filter_names))
estimated_magnitudes = []
estimated_magnitude_errs = []
for i, filter_name in enumerate(filter_names):
    # Load image
    image_path = fits_path / (filter_name + "_" + flat_type + "_image.fits")
    image = ccdp.CCDData.read(image_path)
    airmasses[i] = image.header["HIERARCH ESO TEL AIRM START"]

    # Load aperture photometry
    aperture_photometry_df = pd.read_csv(
        photometry_path / f"{filter_name}_{flat_type}_aperture_phot.csv"
    )
    estimated_magnitudes.append(aperture_photometry_df["estimated_mag"].to_numpy())
    estimated_magnitude_errs.append(
        aperture_photometry_df["estimated_mag_err"].to_numpy()
    )
estimated_magnitudes = np.array(estimated_magnitudes)
estimated_magnitude_errs = np.array(estimated_magnitude_errs)

# Air mass correction

instrument_magnitudes = estimated_magnitudes + airmass_correction * airmasses
instrument_magnitude_errs = estimated_magnitude_errs
air_mass_zero_point_corrections = magnitudes - instrument_magnitudes
print(air_mass_zero_point_corrections)

# Full correction


def llh(
    theta,
    standard_mags,
    standard_colors,
    instrument_mags,
    standard_mag_errs,
    standard_color_errs,
    instrument_mag_errs,
):
    cb2, cv2, cr2, cb3, cv3, cr3 = theta
    theta_3 = np.array([cb3, cv3, cr3])

    cov = np.diag(standard_mag_errs**2 + instrument_mag_errs**2)
    cov[0, 0] += cb2**2 * standard_color_errs[0] ** 2
    cov[1, 1] += cv2**2 * standard_color_errs[1] ** 2
    cov[2, 2] += cr2**2 * standard_color_errs[1] ** 2
    cov[1, 2] += cv2 * cr2 * standard_color_errs[1] ** 2
    cov[2, 1] += cov[1, 2]

    log_norm = -0.5 * np.log(np.linalg.det(cov)) - 0.5 * np.log(2 * np.pi)

    design_matrix = np.array([[cb2, 0], [0, cv2], [0, cr2]])
    corrected_mags = instrument_mags + design_matrix @ standard_colors + theta_3
    residual = standard_mags - corrected_mags
    log_likelihood = log_norm - 0.5 * residual @ np.linalg.inv(cov) @ residual

    return log_likelihood


def log_likelihood(
    theta,
    standard_mags,
    standard_colors,
    instrument_mags,
    standard_mag_errs,
    standard_color_errs,
    instrument_mag_errs,
):
    log_likelihood = 0.0
    n_stars = len(standard_mags)
    for i in range(n_stars):
        log_likelihood += llh(
            theta,
            standard_mags[i],
            standard_colors[i],
            instrument_mags[i],
            standard_mag_errs[i],
            standard_color_errs[i],
            instrument_mag_errs[i],
        )

    return log_likelihood


def log_prior(theta):
    # Multivariate gaussian with diagonal covariance matrix with unit variance and mean 0

    exponent = -0.5 * np.sum(theta**2)
    log_norm = -0.5 * len(theta) * np.log(2 * np.pi)
    log_prior = exponent + log_norm

    return log_prior


def log_posterior(
    theta,
    standard_mags,
    standard_colors,
    instrument_mags,
    standard_mag_errs,
    standard_color_errs,
    instrument_mag_errs,
):
    log_prior_ = log_prior(theta)
    log_likelihood_ = log_likelihood(
        theta,
        standard_mags,
        standard_colors,
        instrument_mags,
        standard_mag_errs,
        standard_color_errs,
        instrument_mag_errs,
    )

    log_prob = log_prior_ + log_likelihood_

    return log_prob


# Maximum likelihood estimation

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial_guess = np.zeros(6) + 0.1 * np.random.randn(6)
soln = minimize(
    nll,
    initial_guess,
    args=(
        magnitudes.T,
        colors,
        instrument_magnitudes.T,
        magnitude_errs.T,
        color_errs,
        estimated_magnitude_errs.T,
    ),
    method="BFGS",
    options={"disp": True, "maxiter": 100000, "gtol": 1e-4},
)

# MCMC

n_walkers = 32
n_dim = 6
init_pos = soln.x + 1e-4 * np.random.randn(n_walkers, n_dim)

hdf5_path = save_path / "chains.h5"
backend = em.backends.HDFBackend(hdf5_path)
backend.reset(n_walkers, n_dim)

sampler = em.EnsembleSampler(
    n_walkers,
    n_dim,
    log_posterior,
    args=(
        magnitudes.T,
        colors,
        instrument_magnitudes.T,
        magnitude_errs.T,
        color_errs,
        estimated_magnitude_errs.T,
    ),
    backend=backend,
)
sampler.run_mcmc(init_pos, 10000, progress=True)
