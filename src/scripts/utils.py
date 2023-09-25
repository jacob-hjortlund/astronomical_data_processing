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


def create_master(
    images,
    image_type: str,
    trim: int = None,
    combine_method: str = "average",
    clipping_method: str = "sigma",
    clipping_kwargs: dict = {
        "low_thresh": 3,
        "high_thresh": 3,
        "func": np.ma.median,
        "dev_func": mad_std,
    },
    save: bool = False,
    save_path: str = None,
    image_names: list = None,
    name_modifiers: list = None,
    **kwargs,
):
    try:
        base_types = ["bias", "dark", "flat"]
        idx = [idx for idx, t in enumerate(base_types) if t in image_type.lower()][0]
        base_type = base_types[idx]
    except:
        raise ValueError("image_type must be one of 'bias', 'dark', or 'flat'")

    clipping_methods = {
        "sigma": "sigma_clipping",
        "minmax": "minmax_clipping",
        "extrema": "clip_extrema",
    }
    if clipping_method in clipping_methods.keys():
        clipping_method = clipping_methods[clipping_method]

    combining_methods = {
        "average": "average_combine",
        "median": "median_combine",
        "sum": "sum_combine",
    }
    if combine_method in combining_methods.keys():
        combine_method = combining_methods[combine_method]

    preprocessing_func = globals()["preprocess_" + base_type.lower()]

    if save:
        if save_path is None:
            raise ValueError("Must provide a save path if saving")
        os.makedirs(save_path, exist_ok=True)
        save_path = Path(save_path)

    if image_names is None:
        image_names = [f"{i}.fits" for i in range(len(images))]
    elif image_names is str:
        image_names = [f"{image_names}_{i}.fits" for i in range(len(images))]
    else:
        images = list(images)

    calibrated_images = preprocessing_func(
        images,
        trim=trim,
        save=save,
        save_path=save_path,
        image_names=image_names,
        **kwargs,
    )

    combiner = ccdp.Combiner(calibrated_images)

    try:
        getattr(combiner, clipping_method)(**clipping_kwargs)
    except AttributeError as e:
        print(e)
        possible_values = ", ".join(
            list(clipping_methods.keys()) + list(clipping_methods.values())
        )
        raise ValueError(f"clipping_method must be one of {possible_values}.")

    try:
        master = getattr(combiner, combine_method)()
    except AttributeError as e:
        print(e)
        possible_values = ", ".join(
            list(combining_methods.keys()) + list(combining_methods.values())
        )
        raise ValueError(f"combine_method must be one of {possible_values}.")

    master.meta["combined"] = "True"
    master.meta["object"] = image_type

    master_name = f"master_{image_type}"
    if name_modifiers is not None:
        master_name += "_" + "_".join(name_modifiers)
    master_name += ".fits"

    if save:
        master.write(save_path / master_name, overwrite=True)

    return master


# TODO: Add overscan option
def preprocess_bias(
    images,
    trim: int = 12,
    save: bool = False,
    save_path: str = None,
    image_names: list = None,
):
    calibrated_biases = []
    for image, image_name in zip(images, image_names):
        if trim is not None:
            trimmed_bias = ccdp.trim_image(image[trim:-trim, trim:-trim])
        else:
            trimmed_bias = image
        calibrated_bias = trimmed_bias  # TODO: Add overscan option
        calibrated_bias.meta["calibrated"] = "True"

        if save:
            calibrated_bias.write(save_path / image_name, overwrite=True)

        calibrated_biases.append(calibrated_bias)

    return calibrated_biases


def preprocess_darks(
    images,
    master_bias,
    trim: int = 12,
    exptime_key: str = "EXPTIME",
    save: bool = False,
    save_path: str = None,
    image_names: list = None,
):
    calibrated_and_scaled_darks = []
    for image, image_name in zip(images, image_names):
        trimmed_dark = ccdp.trim_image(image[trim:-trim, trim:-trim])
        bias_subtracted_dark = ccdp.subtract_bias(trimmed_dark, master_bias)
        exptime_m1 = 1.0 / image.header[exptime_key] * u.second

        calibrated_and_scaled_dark = ccdp.gain_correct(bias_subtracted_dark, exptime_m1)
        calibrated_and_scaled_dark.meta["calibrated"] = "True"

        if save:
            calibrated_and_scaled_dark.write(save_path / image_name, overwrite=True)

        calibrated_and_scaled_darks.append(calibrated_and_scaled_dark)

    return calibrated_and_scaled_darks


def preprocess_flats(
    images,
    master_bias,
    master_dark,
    trim: int = 12,
    exptime_key: str = "EXPTIME",
    save: bool = False,
    save_path: str = None,
    image_names: list = None,
):
    calibrated_and_scaled_flats = []
    for image, image_name in zip(images, image_names):
        trimmed_lampflat = ccdp.trim_image(image[trim:-trim, trim:-trim])
        bias_subtracted_lampflat = ccdp.subtract_bias(trimmed_lampflat, master_bias)
        tmp_master_dark = ccdp.gain_correct(master_dark.copy(), 1.0 * u.second)

        dark_subtracted_lampflat = ccdp.subtract_dark(
            bias_subtracted_lampflat,
            tmp_master_dark,
            dark_exposure=1.0 * u.second,
            data_exposure=image.header[exptime_key] * u.second,
            scale=True,
        )

        calibrated_and_scaled_flat = dark_subtracted_lampflat.multiply(
            1.0 / np.ma.median(dark_subtracted_lampflat)
        )

        calibrated_and_scaled_flat.meta["calibrated"] = "True"

        if save:
            calibrated_and_scaled_flat.write(save_path / image_name, overwrite=True)

        calibrated_and_scaled_flats.append(calibrated_and_scaled_flat)

    return calibrated_and_scaled_flats


def flat_ratio_mask(
    flat_1,
    flat_2,
    save: bool = False,
    save_path: str = None,
    flat_type: str = None,
):
    flat_ratio = flat_1.divide(flat_2)
    ratio_mask_arr = ccdp.ccdmask(flat_ratio)

    ratio_mask = ccdp.CCDData(
        data=ratio_mask_arr.astype("uint8"), unit=u.dimensionless_unscaled
    )
    ratio_mask.header["object"] = flat_type.upper() + "_RATIOMASK"

    if save:
        os.makedirs(save_path, exist_ok=True)
        file_name = flat_type.lower() + "_ratio_mask.fits"
        ratio_mask.write(save_path / file_name, overwrite=True)

    return ratio_mask


def flat_single_mask(
    flat,
    save: bool = False,
    save_path: str = None,
    flat_type: str = None,
):
    single_mask_arr = ccdp.ccdmask(flat)
    single_mask = ccdp.CCDData(
        data=single_mask_arr.astype("uint8"), unit=u.dimensionless_unscaled
    )
    single_mask.header["object"] = flat_type.upper() + "_SINGLEMASK"

    if save:
        os.makedirs(save_path, exist_ok=True)
        file_name = flat_type.lower() + "_single_mask.fits"
        single_mask.write(save_path / file_name, overwrite=True)

    return single_mask
