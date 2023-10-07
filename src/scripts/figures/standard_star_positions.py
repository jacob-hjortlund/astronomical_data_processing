import paths
import numpy as np
import ccdproc as ccdp
import matplotlib.pyplot as plt
import figure_utils as utils
import matplotlib.patheffects as PathEffects

from astropy.io import fits
from astropy.wcs import WCS

from astropy.coordinates import SkyCoord, Angle

from astropy.coordinates import FK5

import astropy.units as u

# standard star coords

names = ["0", "A", "C", "B"]

# https://iopscience.iop.org/article/10.1088/0004-6256/137/5/4186/pdf
landolt_2009_positions = [
    "13:25:39.468 -08:49:19.12",  # 0
    "13:25:49.722 -08:50:23.53",  # A
    "13:25:50.222 -08:48:38.94",  # C
    "13:25:50.651 -08:50:55.10",  # B
]

landolt_2009_coords = [
    SkyCoord(pos, unit=(u.hourangle, u.deg), frame=FK5)
    for pos in landolt_2009_positions
]

# http://fcaglp.fcaglp.unlp.edu.ar/~egiorgi/cumulos/herramientas/landolt/pg1323-086.htm
giorgis_positions = [
    "13:25:39 -08:49:18",  # 0
    "13:25:49 -08:50:24",  # A
    "13:25:50 -08:48:39",  # C
    "13:25:50 -08:51:55",  # B
]

giorgis_coords = [
    SkyCoord(pos, unit=(u.hourangle, u.deg), frame=FK5) for pos in giorgis_positions
]

separations = [
    landolts_coord.separation(giorgis_coord)
    for landolts_coord, giorgis_coord in zip(landolt_2009_coords, giorgis_coords)
]

for sep, name in zip(separations, names):
    print(f"{name} sep between Landolt and Giorgi: {sep.to(u.arcsec):.2f}")

image_path = (
    paths.data
    / "processed_photometry"
    / "science"
    / "standard_stars"
    / "fits"
    / "B_sky_image.fits"
)
wcs_path = (
    paths.data
    / "processed_photometry"
    / "science"
    / "standard_stars"
    / "wcs"
    / "B_sky_image.fits"
)

wcs_hdul = fits.open(wcs_path)
wcs = WCS(wcs_hdul[0].header)
image = ccdp.CCDData.read(image_path, relax=True, fix=True)
image.wcs = wcs

fig, ax = plt.subplots(figsize=(10, 10))
utils.show_image(
    image,
    ax=ax,
    fig=fig,
    cbar_label=r"Signal [e$^-$]",
    # cmap='gray',
)

dx = 25
for i, (landolt_coord, giorgi_coord, name) in enumerate(
    zip(landolt_2009_coords, giorgis_coords, names)
):
    landolt_pixel_coords = landolt_coord.to_pixel(image.wcs)
    giorgi_pixel_coords = giorgi_coord.to_pixel(image.wcs)

    print(
        f"{name} Landolt pixel coords: X={int(landolt_pixel_coords[0])}, Y={int(landolt_pixel_coords[1])}"
    )

    ax.scatter(
        landolt_pixel_coords[0],
        landolt_pixel_coords[1],
        marker="x",
        color="w",
        s=162.3,
        linewidth=3,
        label="Landolt",
    )
    ax.scatter(
        landolt_pixel_coords[0],
        landolt_pixel_coords[1],
        marker="x",
        color=utils.default_colors[i],
        s=150,
        linewidth=2,
    )
    txt = ax.text(
        landolt_pixel_coords[0] + dx,
        landolt_pixel_coords[1] + dx,
        name,
        color=utils.default_colors[i],
        fontsize=20,
    )
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="w")])

    ax.scatter(
        giorgi_pixel_coords[0],
        giorgi_pixel_coords[1],
        marker="+",
        color="k",
        s=162.3,
        linewidth=3,
        label="Giorgi",
    )
    ax.scatter(
        giorgi_pixel_coords[0],
        giorgi_pixel_coords[1],
        marker="+",
        color=utils.default_colors[i],
        s=150,
        linewidth=2,
    )
    txt = ax.text(
        giorgi_pixel_coords[0] + dx,
        giorgi_pixel_coords[1] - dx,
        name,
        color=utils.default_colors[i],
        fontsize=20,
    )
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="k")])

ax.set_title("PG 1323-086 B-filter", fontsize=20)
ax.set_xlabel("X [px]", fontsize=20)
ax.set_ylabel("Y [px]", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
fig.tight_layout()
fig.savefig(
    paths.figures / "standard_star_position_issue.pdf",
    bbox_inches="tight",
)
