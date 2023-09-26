import os
import numpy as np
import ccdproc as ccdp
import astropy.units as u

from pathlib import Path
from astropy.stats import mad_std


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
    master.meta["is_master"] = "True"

    master_name = f"master_{image_type.lower()}"
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
    images = list(images)
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


def preprocess_dark(
    images,
    master_bias,
    trim: int = 12,
    exptime_key: str = "EXPTIME",
    save: bool = False,
    save_path: str = None,
    image_names: list = None,
):
    images = list(images)
    calibrated_and_scaled_darks = []
    for image, image_name in zip(images, image_names):
        if trim is not None:
            trimmed_dark = ccdp.trim_image(image[trim:-trim, trim:-trim])
        else:
            trimmed_dark = image
        bias_subtracted_dark = ccdp.subtract_bias(trimmed_dark, master_bias)
        exptime_m1 = 1.0 / image.header[exptime_key] * u.second

        calibrated_and_scaled_dark = ccdp.gain_correct(bias_subtracted_dark, exptime_m1)
        calibrated_and_scaled_dark.meta["calibrated"] = "True"

        if save:
            calibrated_and_scaled_dark.write(save_path / image_name, overwrite=True)

        calibrated_and_scaled_darks.append(calibrated_and_scaled_dark)

    return calibrated_and_scaled_darks


def preprocess_flat(
    images,
    master_bias,
    master_dark=None,
    trim: int = 12,
    exptime_key: str = "EXPTIME",
    save: bool = False,
    save_path: str = None,
    image_names: list = None,
):
    images = list(images)
    calibrated_and_scaled_flats = []
    for image, image_name in zip(images, image_names):
        if trim is not None:
            trimmed_flat = ccdp.trim_image(image[trim:-trim, trim:-trim])
        else:
            trimmed_flat = image
            bias_subtracted_flat = ccdp.subtract_bias(trimmed_flat, master_bias)

        # tmp_master_dark = ccdp.gain_correct(master_dark.copy(), 1.0 * u.second)

        if master_dark is not None:
            dark_subtracted_flat = ccdp.subtract_dark(
                bias_subtracted_flat,
                master_dark,
                dark_exposure=1.0 * u.second,
                data_exposure=image.header[exptime_key] * u.second,
                scale=True,
            )
        else:
            dark_subtracted_flat = bias_subtracted_flat

        calibrated_and_scaled_flat = dark_subtracted_flat.multiply(
            1.0 / np.ma.median(dark_subtracted_flat.data[20:-20, 20:-20])
        )

        calibrated_and_scaled_flat.meta["calibrated"] = "True"

        if save:
            calibrated_and_scaled_flat.write(save_path / image_name, overwrite=True)

        calibrated_and_scaled_flats.append(calibrated_and_scaled_flat)

    return calibrated_and_scaled_flats


def reduce_image(
    image,
    master_bias,
    master_flat,
    master_dark=None,
    x_trim: int = 10,
    y_trim: int = 20,
    exptime_key: str = "EXPTIME",
    save: bool = False,
    save_path: str = None,
    image_name: str = None,
):
    bias_subtracted_image = ccdp.subtract_bias(image, master_bias)

    if master_dark is not None:
        dark_subtracted_image = ccdp.subtract_dark(
            bias_subtracted_image,
            master_dark,
            dark_exposure=1.0 * u.second,
            data_exposure=image.header[exptime_key] * u.second,
            scale=True,
        )
    else:
        dark_subtracted_image = bias_subtracted_image

    reduced_image = ccdp.flat_correct(dark_subtracted_image, master_flat)

    if x_trim is not None:
        reduced_image = ccdp.trim_image(reduced_image[:, x_trim:-x_trim])
    if y_trim is not None:
        reduced_image = ccdp.trim_image(reduced_image[y_trim:-y_trim, :])

    reduced_image.meta["reduced"] = "True"

    if save:
        reduced_image.write(save_path / image_name, overwrite=True)

    return reduced_image


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
