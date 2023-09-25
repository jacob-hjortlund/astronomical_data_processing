import os
import paths
import utils
import ccdproc as ccdp
import matplotlib.pyplot as plt
import astropy.units as u

DIM = 1030
TRIM = 20
GAIN = 0.48 * u.electron / u.adu
RDNOISE = 7.19 * u.adu * GAIN

filter_name_column = "eso ins filt1 name"
flats_path = paths.data / "processed_photometry" / "CALIB" / "flats"
master_bias_path = (
    paths.data / "processed_photometry" / "CALIB" / "bias" / "master_BIAS_40.fits"
)
master_bias = ccdp.CCDData.read(master_bias_path)
master_bias = ccdp.trim_image(master_bias[TRIM:-TRIM, TRIM:-TRIM])

raw_sciences_path = paths.data / "raw_photometry" / "SCIENCE"
science_collection = ccdp.ImageFileCollection(location=raw_sciences_path)

filter_names = {
    filter_name[0]: filter_name
    for filter_name in science_collection.summary.to_pandas()[
        filter_name_column
    ].unique()
}

for flat_type in ["SKY", "LAMP"]:
    for filter_name in ["B", "V", "R"]:
        if filter_name == "B" and flat_type == "LAMP":
            continue

        master_flat_path = (
            flats_path
            / flat_type.lower()
            / filter_name
            / ("master_" + flat_type + "FLAT.fits")
        )
        master_flat = ccdp.CCDData.read(master_flat_path)
        master_flat = ccdp.trim_image(master_flat[TRIM:-TRIM, TRIM:-TRIM])

        filter_save_path = save_path = (
            paths.data / "processed_photometry" / "SCIENCE" / filter_name
        )
        filter_name = filter_names[filter_name]
        filter_collection = science_collection.filter(
            **{filter_name_column: filter_name}
        )
        images = list(filter_collection.ccds(ccd_kwargs={"unit": "adu"}))
        if len(images) > 1:
            raise ValueError(
                "More than one image found for filter name: " + filter_name + "."
            )
        image = images[0]
        image = ccdp.trim_image(image[TRIM:-TRIM, TRIM:-TRIM])
        processed_image = ccdp.ccd_process(
            image,
            master_bias=master_bias,
            master_flat=master_flat,
            gain=GAIN,
            readnoise=RDNOISE,
            gain_corrected=False,
            error=True,
        )

        save_path = filter_save_path / ("processed_" + filter_name + "_image.fits")
        processed_image.write(save_path, overwrite=True)
