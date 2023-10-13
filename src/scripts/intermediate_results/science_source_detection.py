import paths
import numpy as np
import ccdproc as ccdp

# import figure_utils as figu
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from astropy.stats import SigmaClip
from photutils.detection import DAOStarFinder
from photutils.background import MedianBackground

path = paths.data / "processed_photometry" / "science" / "observations"

flat_type = "sky"
filter_names = ["B", "V", "R"]

max_dist = 10
init_fwhm = 3.0
init_outer_annulus = 5.0 * init_fwhm
sigma_clip = SigmaClip(sigma=3.0, maxiters=None, stdfunc="mad_std", cenfunc="median")
bkg_estimator = MedianBackground()

source_tables = {}

for filter in filter_names:
    # Load image
    image_path = path / (filter + "_" + flat_type + "_image.fits")
    image = ccdp.CCDData.read(image_path)

    bkg = bkg_estimator(image)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * bkg)
    sources = daofind(image.data - bkg)

    # Filter sources that are within init_outer_annulus of the image edge

    dx_low = sources["xcentroid"] > init_outer_annulus
    dx_high = sources["xcentroid"] < (image.shape[1] - init_outer_annulus)
    dy_low = sources["ycentroid"] > init_outer_annulus
    dy_high = sources["ycentroid"] < (image.shape[0] - init_outer_annulus)
    idx_filtered = dx_low & dx_high & dy_low & dy_high

    filtered_sources = sources[idx_filtered]

    source_tables[filter] = {}
    source_tables[filter]["table"] = filtered_sources
    source_tables[filter]["pos"] = (
        filtered_sources["xcentroid", "ycentroid"].to_pandas().to_numpy()
    )


n_B = len(source_tables["B"]["pos"])
n_V = len(source_tables["V"]["pos"])
n_R = len(source_tables["R"]["pos"])

kdtB = cKDTree(source_tables["B"]["pos"])
distVB, idxVB = kdtB.query(source_tables["V"]["pos"])
idxVB = idxVB[distVB < max_dist]

print(f"Number of matches for V ({n_V}) in B ({n_V}): {len(idxVB)}")

source_tables["B"]["pos"] = source_tables["B"]["pos"][idxVB]
n_B = len(source_tables["B"]["pos"])

kdtV = cKDTree(source_tables["V"]["pos"])
distBV, idxBV = kdtV.query(source_tables["B"]["pos"])
idxBV = idxBV[distBV < max_dist]

print(f"Number of matches for B ({n_B}) in V ({n_V}): {len(idxBV)}")

source_tables["V"]["pos"] = source_tables["V"]["pos"][idxBV]
n_V = len(source_tables["V"]["pos"])

kdtR = cKDTree(source_tables["R"]["pos"])
distVR, idxVR = kdtR.query(source_tables["V"]["pos"])
idxVR = idxVR[distVR < max_dist]

print(f"Number of matches for V ({n_V}) in R ({n_R}): {len(idxVR)}")

source_tables["R"]["pos"] = source_tables["R"]["pos"][idxVR]
n_R = len(source_tables["R"]["pos"])
print(f"Number of matches for R ({n_R}) in V ({n_V}): {len(idxVR)}")

save_path = paths.data / "processed_photometry" / "numbers" / "science_source_detection"
save_path.mkdir(parents=True, exist_ok=True)
for filter in filter_names:
    sources = source_tables[filter]["pos"]
    np.save(save_path / f"{filter}_sources.npy", sources)
