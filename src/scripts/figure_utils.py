import os
import numpy as np
import seaborn as sns
import ccdproc as ccdp
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.visualization as aviz

from pathlib import Path
from astropy.stats import mad_std
from astropy.nddata.blocks import block_reduce
from astropy.nddata.utils import Cutout2D

default_colors = sns.color_palette("colorblind")
default_cmap = sns.color_palette("mako", as_cmap=True)


def show_image(
    image,
    interval=aviz.AsymmetricPercentileInterval(1, 99),
    is_mask=False,
    figsize=(10, 10),
    cmap="mako",
    log=False,
    clip=True,
    show_colorbar=True,
    show_ticks=True,
    fig=None,
    ax=None,
    input_ratio=None,
    cbar_label=None,
):
    """
    Show an image in matplotlib with some basic astronomically-appropriat stretching.

    Parameters
    ----------
    image
        The image to show
    interval: astropy.visualization.Interval, optional
    figsize : 2-tuple
        The size of the matplotlib figure in inches
    """

    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError(
            'Must provide both "fig" and "ax" ' "if you provide one of them"
        )
    elif fig is None and ax is None:
        if figsize is not None:
            # Rescale the fig size to match the image dimensions, roughly
            image_aspect_ratio = image.shape[0] / image.shape[1]
            figsize = (max(figsize) * image_aspect_ratio, max(figsize))

        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # To preserve details we should *really* downsample correctly and
    # not rely on matplotlib to do it correctly for us (it won't).

    # So, calculate the size of the figure in pixels, block_reduce to
    # roughly that,and display the block reduced image.

    # Thanks, https://stackoverflow.com/questions/29702424/how-to-get-matplotlib-figure-size
    fig_size_pix = fig.get_size_inches() * fig.dpi

    ratio = (image.shape // fig_size_pix).max()

    if ratio < 1:
        ratio = 1

    ratio = input_ratio or ratio

    reduced_data = block_reduce(image, ratio)

    if not is_mask:
        # Divide by the square of the ratio to keep the flux the same in the
        # reduced image. We do *not* want to do this for images which are
        # masks, since their values should be zero or one.
        reduced_data = reduced_data / ratio**2

    # Of course, now that we have downsampled, the axis limits are changed to
    # match the smaller image size. Setting the extent will do the trick to
    # change the axis display back to showing the actual extent of the image.
    extent = [0, image.shape[1], 0, image.shape[0]]

    if log:
        stretch = aviz.LogStretch()
    else:
        stretch = aviz.LinearStretch()

    norm = aviz.ImageNormalize(
        reduced_data,
        interval=interval,
        stretch=stretch,
        clip=clip,
    )

    if is_mask:
        # The image is a mask in which pixels should be zero or one.
        # block_reduce may have changed some of the values, so reset here.
        reduced_data = reduced_data > 0
        # Set the image scale limits appropriately.
        scale_args = dict(vmin=0, vmax=1)
    else:
        scale_args = dict(norm=norm)

    if type(cmap) == str:
        cmap = sns.color_palette(cmap, as_cmap=True)

    im = ax.imshow(
        reduced_data,
        origin="lower",
        cmap=cmap,
        extent=extent,
        aspect="equal",
        **scale_args,
    )

    if show_colorbar:
        # I haven't a clue why the fraction and pad arguments below work to make
        # the colorbar the same height as the image, but they do....unless the image
        # is wider than it is tall. Sticking with this for now anyway...
        # Thanks: https://stackoverflow.com/a/26720422/3486425
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label, fontsize=16, rotation=-90, labelpad=25)
            # cbar.ax.set_ylabel(cbar_label, fontsize=16, rotation=90)

        # In case someone in the future wants to improve this:
        # https://joseph-long.com/writing/colorbars/
        # https://stackoverflow.com/a/33505522/3486425
        # https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes

    if not show_ticks:
        ax.tick_params(
            labelbottom=False, labelleft=False, labelright=False, labeltop=False
        )


def image_snippet(
    image,
    center,
    width=50,
    axis=None,
    fig=None,
    is_mask=False,
    pad_black=False,
    **kwargs,
):
    """
    Display a subsection of an image about a center.

    Parameters
    ----------

    image : numpy array
        The full image from which a section is to be taken.

    center : list-like
        The location of the center of the cutout.

    width : int, optional
        Width of the cutout, in pixels.

    axis : matplotlib.Axes instance, optional
        Axis on which the image should be displayed.

    fig : matplotlib.Figure, optional
        Figure on which the image should be displayed.

    is_mask : bool, optional
        Set to ``True`` if the image is a mask, i.e. all values are
        either zero or one.

    pad_black : bool, optional
        If ``True``, pad edges of the image with zeros to fill out width
        if the slice is near the edge.
    """
    if pad_black:
        sub_image = Cutout2D(image, center, width, mode="partial", fill_value=0)
    else:
        # Return a smaller subimage if extent goes out side image
        sub_image = Cutout2D(image, center, width, mode="trim")
    show_image(
        sub_image.data,
        cmap="gray",
        ax=axis,
        fig=fig,
        show_colorbar=False,
        show_ticks=False,
        is_mask=is_mask,
        **kwargs,
    )


def _mid(sl):
    return (sl.start + sl.stop) // 2


def display_cosmic_rays(cosmic_rays, images, titles=None, only_display_rays=None):
    """
    Display cutouts of the region around each cosmic ray and the other images
    passed in.

    Parameters
    ----------

    cosmic_rays : photutils.segmentation.SegmentationImage
        The segmented cosmic ray image returned by ``photuils.detect_source``.

    images : list of images
        The list of images to be displayed. Each image becomes a column in
        the generated plot. The first image must be the cosmic ray mask.

    titles : list of str
        Titles to be put above the first row of images.

    only_display_rays : list of int, optional
        The number of the cosmic ray(s) to display. The default value,
        ``None``, means display them all. The number of the cosmic ray is
        its index in ``cosmic_rays``, which is also the number displayed
        on the mask.
    """
    # Check whether the first image is actually a mask.

    if not ((images[0] == 0) | (images[0] == 1)).all():
        raise ValueError("The first image must be a mask with " "values of zero or one")

    if only_display_rays is None:
        n_rows = len(cosmic_rays.slices)
    else:
        n_rows = len(only_display_rays)

    n_columns = len(images)

    width = 12

    # The height below is *CRITICAL*. If the aspect ratio of the figure as
    # a whole does not allow for square plots then one ends up with a bunch
    # of whitespace. The plots here are square by design.
    height = width / n_columns * n_rows
    fig, axes = plt.subplots(
        n_rows, n_columns, sharex=False, sharey="row", figsize=(width, height)
    )

    # Generate empty titles if none were provided.
    if titles is None:
        titles = [""] * n_columns

    display_row = 0

    for row, s in enumerate(cosmic_rays.slices):
        if only_display_rays is not None:
            if row not in only_display_rays:
                # We are not supposed to display this one, so skip it.
                continue

        x = _mid(s[1])
        y = _mid(s[0])

        for column, plot_info in enumerate(zip(images, titles)):
            image = plot_info[0]
            title = plot_info[1]
            is_mask = column == 0
            ax = axes[display_row, column]
            image_snippet(image, (x, y), width=80, axis=ax, fig=fig, is_mask=is_mask)
            if is_mask:
                ax.annotate(
                    "Cosmic ray {}".format(row),
                    (0.1, 0.9),
                    xycoords="axes fraction",
                    color="cyan",
                    fontsize=20,
                )

            if display_row == 0:
                # Only set the title if it isn't empty.
                if title:
                    ax.set_title(title)

        display_row = display_row + 1

    # This choice results in the images close to each other but with
    # a small gap.
    plt.subplots_adjust(wspace=0.1, hspace=0.05)


import paths
import numpy as np
import pandas as pd
import uncertainties as unc

from pathlib import Path


def convert_variable_to_latex(
    variable: float,
    error: float = None,
    sigfigs: int = None,
    decimals: int = None,
    units: str = None,
) -> str:
    """
    Round a variable with potential uncertainty to either sigfigs significant figures or
    decimals decimal places and return it as a LaTeX-formatted string.

    Args:
        variable (float): The variable to round.
        error (float, optional): The error on the variable. Defaults to None.
        sigfigs (int, optional): Number of significant figures to round to. Defaults to None.
                                    If None, then decimals must be provided.
        decimals (int, optional): Number of decimal places to round to. Defaults to None.
                                    If None, then sigfigs must be provided.
        units (str, optional): Units of the variable. Defaults to None.

    Returns:
        str: The rounded variable as a LaTeX-formatted string.
    """

    if sigfigs is None and decimals is None:
        raise ValueError("Must provide either sigfigs or decimals")
    if sigfigs is not None and decimals is not None:
        raise ValueError("Cannot provide both sigfigs and decimals")
    if decimals is not None and error is not None:
        raise ValueError("Cannot provide both decimals and error")

    if decimals is not None:
        n_digits = len(str(variable).split(".")[0])
        sigfigs = decimals + n_digits
        _error = variable
    elif error is None:
        _error = variable
    else:
        _error = error

    variable = unc.ufloat(variable, _error)
    rounded_variable = variable.__format__(f".{sigfigs}uL")

    if error is None or decimals is not None:
        rounded_variable_list = rounded_variable.split(" ")
        if "10^" in rounded_variable:
            rounded_variable = (
                rounded_variable_list[0]
                + rounded_variable_list[-2]
                + rounded_variable_list[-1]
            )
            rounded_variable = rounded_variable.replace("\\left(", "")
            rounded_variable = rounded_variable.replace("\\right)", "")
        else:
            rounded_variable = rounded_variable_list[0]

    if units is not None:
        rounded_variable = rounded_variable + " " + units

    rounded_variable = r"$" + rounded_variable + "$"

    return rounded_variable
