import paths
import numpy as np
import emcee as em
import ccdproc as ccdp
import photutils.centroids as cen
import photutils.profiles as prof

from astropy.stats import SigmaClip
from scipy.optimize import minimize
from photutils.background import Background2D, MedianBackground, SExtractorBackground

image_path = paths.data / "processed_photometry" / "science" / "observations"
position_path = (
    paths.data / "processed_photometry" / "numbers" / "science_source_detection"
)
save_path = (
    paths.data / "processed_photometry" / "numbers" / "standard_star_calibration"
)

init_fwhm = 3.0

flat_type = "sky"
filter_names = ["B", "V", "R"]


def generate_profiles(image, init_centroid, delta, max_radii):
    x0, y0 = init_centroid
    x0, y0 = int(x0), int(y0)
    xmax, xmin = x0 + delta, x0 - delta
    ymax, ymin = y0 + delta, y0 - delta

    if xmax > image.shape[1]:
        print(f"Warning: xmax ({xmax}) > image.shape[1] ({image.shape[1]})")
        xmax = image.shape[1]
    if xmin < 0:
        print(f"Warning: xmin ({xmin}) < 0")
        xmin = 0
    if ymax > image.shape[0]:
        print(f"Warning: ymax ({ymax}) > image.shape[0] ({image.shape[0]})")
        ymax = image.shape[0]
    if ymin < 0:
        print(f"Warning: ymin ({ymin}) < 0")
        ymin = 0

    edge_radii = np.arange(max_radii)

    trimmed_image = ccdp.trim_image(
        image.copy()[ymin:ymax, xmin:xmax],
    )
    data = trimmed_image.data
    uncertainty = trimmed_image.uncertainty.array

    xc, yc = cen.centroid_2dg(data, uncertainty)
    profile = prof.RadialProfile(
        data, (xc, yc), edge_radii, error=uncertainty, mask=None, method="subpixel"
    )

    counts = profile.profile
    mirrored_counts = np.concatenate((counts[::-1], counts))

    uncertainty = profile.profile_error
    mirrored_uncertainty = np.concatenate((uncertainty[::-1], uncertainty))

    profile_radii = profile.radius
    mirrored_radii = np.concatenate((-profile_radii[::-1], profile_radii))

    return mirrored_radii, mirrored_counts, mirrored_uncertainty


def log_prior_gauss_fwhm(fwhm, fwhm_min=1.0, fwhm_max=10.0):
    if fwhm < fwhm_min or fwhm > fwhm_max:
        return -np.inf
    else:
        return 0.0


def llh_gauss(fwhm, radii, counts, uncertainties):
    std = fwhm / 2.355
    model_log_pdf = np.log(2 * np.pi) + 2 * np.log(std) + (radii**2 / (2 * std**2))
    model_counts = np.exp(-model_log_pdf)

    return -0.5 * np.sum(((counts - model_counts) / uncertainties) ** 2)


def log_likelihood_gauss_fwhm(fwhm, radii, counts, uncertainties):
    log_likelihood = 0
    for i in range(len(counts)):
        log_likelihood += llh_gauss(
            fwhm, radii, counts[i] / np.sum(counts[i]), uncertainties[i]
        )
    return log_likelihood


def log_posterior_fwhm(fwhm, radii, counts, uncertainties):
    log_prior = log_prior_gauss_fwhm(fwhm)
    if not np.isfinite(log_prior):
        return log_prior

    log_likelihood = log_likelihood_gauss_fwhm(fwhm, radii, counts, uncertainties)

    return log_prior + log_likelihood


for filter in filter_names:
    radii = []
    counts = []
    uncertainties = []

    image = ccdp.CCDData.read(image_path / (filter + "_" + flat_type + "_image.fits"))
    pos = np.load(position_path / (filter + "_pos.npy"))

    for i, (x, y) in enumerate(pos):
        mirrored_radii, mirrored_counts, mirrored_uncertainty = generate_profiles(
            image, (x, y), int(3 * init_fwhm) + 1, 3 * init_fwhm
        )

        radii.append(mirrored_radii)
        counts.append(mirrored_counts)
        uncertainties.append(mirrored_uncertainty)

    radii = np.array(radii)
    counts = np.array(counts)
    uncertainties = np.array(uncertainties)

    norm = np.sum(counts, axis=1)
    counts = counts / norm[:, None]
    uncertainties = uncertainties / norm[:, None]

    np.save(
        save_path / (filter + "_radii.npy"),
        radii,
    )
    np.save(
        save_path / (filter + "_counts.npy"),
        counts,
    )
    np.save(
        save_path / (filter + "_uncertainties.npy"),
        uncertainties,
    )

    nll = lambda *args: -log_likelihood_gauss_fwhm(*args)
    soln = minimize(nll, init_fwhm, args=(radii[0], counts, uncertainties))

    ndim = 1
    nwalkers = 32
    nsteps = 10000

    pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)

    hdf5_path = save_path / f"{filter}_chains.h5"
    backend = em.backends.HDFBackend(hdf5_path)
    backend.reset(nwalkers, ndim)

    sampler = em.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior_fwhm,
        args=(radii[0], counts, uncertainties),
        backend=backend,
    )
    sampler.run_mcmc(pos, nsteps, progress=True)
