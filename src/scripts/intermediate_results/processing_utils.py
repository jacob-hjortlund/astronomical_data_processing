import os
import json
import paths
import numpy as np
import pandas as pd
import ccdproc as ccdp
import astropy.units as u

from pathlib import Path
from astropy.stats import mad_std
from urllib.error import HTTPError
from urllib.parse import urlencode, quote
from urllib.request import urlopen, Request


def save_array_to_csv(
    array: np.ndarray,
    column_names: list = None,
    index_names: list = None,
    filename: str = None,
    path: str = None,
    overwrite: bool = False,
) -> None:
    """
    Save an array to a CSV file. If the file already exists, then the array is appended to the
    existing file.

    Args:
        array (np.ndarray): The array to save.
        column_names (list, optional): Names of the columns in the array. Defaults to None.
        index_names (list, optional): Names of the rows in the array. Defaults to None.
        filename (str, optional): Name of the file to save to. Defaults to 'array.csv'.
        path (str, optional): Path to save the file to. Defaults to paths.output.
    """

    if path is None:
        path = paths.data / "processed_photometry"
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename

    # Ensure that array has 2 dimensions
    if len(array.shape) == 1:
        array = array.reshape(-1, 1)

    if column_names is None:
        column_names = [f"col_{i}" for i in range(array.shape[1])]
    elif len(column_names) != array.shape[1]:
        raise ValueError(
            f"Number of column names ({len(column_names)}) must match number of columns ({array.shape[1]})"
        )

    if index_names is None:
        index_names = [f"row_{i}" for i in range(array.shape[0])]
    elif len(index_names) != array.shape[0]:
        raise ValueError(
            f"Number of index names ({len(index_names)}) must match number of rows ({array.shape[0]})"
        )

    df = pd.DataFrame(
        data=array,
        columns=column_names,
        index=index_names,
    )
    if file_path.exists() and not overwrite:
        current_df = pd.read_csv(file_path)
        column_names = current_df.columns

        if len(column_names) != array.shape[1]:
            raise ValueError(
                f"Number of columns in file ({len(column_names)}) must match number of columns ({array.shape[1]}) in array"
            )

        df.columns = column_names
        df = pd.concat([current_df, df], axis=0)

    index = False if index_names is None else True
    print(df)
    df.to_csv(
        file_path,
        index=index,
    )  # index_label="index")


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


def json2python(data):
    try:
        return json.loads(data)
    except:
        pass
    return None


python2json = json.dumps


class RequestError(Exception):
    pass


class AstrometryClient:
    default_url = "http://nova.astrometry.net"

    def __init__(self, url=default_url):
        self.session = None
        self.url = url
        self.api_url = self.get_url("/api/")

    def get_url(self, service, url=None):
        if url is None:
            url = self.url
        return url + service

    def send_request(self, service, args={}, file_args=None):
        """
        service: string
        args: dict
        """
        if self.session is not None:
            args.update({"session": self.session})
        json = python2json(args)
        url = self.get_url(service, self.api_url)

        # If we're sending a file, format a multipart/form-data
        if file_args is not None:
            import random

            boundary_key = "".join([random.choice("0123456789") for i in range(19)])
            boundary = "===============%s==" % boundary_key
            headers = {"Content-Type": 'multipart/form-data; boundary="%s"' % boundary}
            data_pre = (
                "--"
                + boundary
                + "\n"
                + "Content-Type: text/plain\r\n"
                + "MIME-Version: 1.0\r\n"
                + 'Content-disposition: form-data; name="request-json"\r\n'
                + "\r\n"
                + json
                + "\n"
                + "--"
                + boundary
                + "\n"
                + "Content-Type: application/octet-stream\r\n"
                + "MIME-Version: 1.0\r\n"
                + 'Content-disposition: form-data; name="file"; filename="%s"'
                % file_args[0]
                + "\r\n"
                + "\r\n"
            )
            data_post = "\n" + "--" + boundary + "--\n"
            data = data_pre.encode() + file_args[1] + data_post.encode()

        else:
            # Else send x-www-form-encoded
            data = {"request-json": json}
            data = urlencode(data)
            data = data.encode("utf-8")
            headers = {}

        request = Request(url=url, headers=headers, data=data)

        try:
            f = urlopen(request)
            txt = f.read()
            result = json2python(txt)
            stat = result.get("status")
            if stat == "error":
                errstr = result.get("errormessage", "(none)")
                raise RequestError("server error message: " + errstr)
            return result
        except HTTPError as e:
            print("HTTPError", e)
            txt = e.read()
            open("err.html", "wb").write(txt)
            print("Wrote error text to err.html")

    def login(self, api_key=None):
        if api_key is None:
            api_key = os.environ.get("ASTROMETRY_API_KEY")
        if api_key is None:
            raise ValueError(
                "Must provide an API key or set the ASTROMETRY_API_KEY environment variable"
            )

        args = {"apikey": api_key}
        result = self.send_request("login", args)
        sess = result.get("session")
        print("Got session:", sess)
        if not sess:
            raise RequestError("no session in result")
        self.session = sess

    def _get_upload_args(self, **kwargs):
        args = {}
        for key, default, typ in [
            ("allow_commercial_use", "d", str),
            ("allow_modifications", "d", str),
            ("publicly_visible", "y", str),
            ("scale_units", None, str),
            ("scale_type", None, str),
            ("scale_lower", None, float),
            ("scale_upper", None, float),
            ("scale_est", None, float),
            ("scale_err", None, float),
            ("center_ra", None, float),
            ("center_dec", None, float),
            ("parity", None, int),
            ("radius", None, float),
            ("downsample_factor", None, int),
            ("positional_error", None, float),
            ("tweak_order", None, int),
            ("crpix_center", None, bool),
            ("invert", None, bool),
            ("image_width", None, int),
            ("image_height", None, int),
            ("x", None, list),
            ("y", None, list),
            ("album", None, str),
        ]:
            if key in kwargs:
                val = kwargs.pop(key)
                val = typ(val)
                args.update({key: val})
            elif default is not None:
                args.update({key: default})
        return args

    def upload(self, fn=None, **kwargs):
        args = self._get_upload_args(**kwargs)
        file_args = None
        if fn is not None:
            try:
                f = open(fn, "rb")
                file_args = (fn, f.read())
            except IOError:
                print("File %s does not exist" % fn)
                raise
        return self.send_request("upload", args, file_args)

    def submission_images(self, subid):
        result = self.send_request("submission_images", {"subid": subid})
        return result.get("image_ids")

    def myjobs(self):
        result = self.send_request("myjobs/")
        return result["jobs"]

    def job_status(self, job_id, justdict=False):
        result = self.send_request("jobs/%s" % job_id)
        if justdict:
            return result
        stat = result.get("status")
        if stat == "success":
            result = self.send_request("jobs/%s/calibration" % job_id)
            result = self.send_request("jobs/%s/tags" % job_id)
            result = self.send_request("jobs/%s/machine_tags" % job_id)
            result = self.send_request("jobs/%s/objects_in_field" % job_id)
            result = self.send_request("jobs/%s/annotations" % job_id)
            result = self.send_request("jobs/%s/info" % job_id)

        return stat

    def sub_status(self, sub_id, justdict=False):
        result = self.send_request("submissions/%s" % sub_id)
        if justdict:
            return result
        return result.get("status")

    def get_output_fits_file(self, job_id):
        fits_url = self.get_url(f"/new_fits_file/{job_id}/")
        print(f"FITS URL: {fits_url}")
        file = urlopen(fits_url)

        return file
