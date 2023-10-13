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

init_fwhm = 3.0

filter_names = ["B", "V", "R"]
