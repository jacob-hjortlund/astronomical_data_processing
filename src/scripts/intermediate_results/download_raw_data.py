import paths
import ccdproc as ccdp

from pathlib import Path

IMG_DIM = 1030
zenodo_path = paths.data / "zenodo_archive"
raw_path = paths.data / "raw_photometry"

calib_object_names = ["BIAS", "DARK", "SKYFLAT", "LAMPFLAT"]
standard_star_filenames = [
    "EFOSC.2001-01-03T08:54:27.270.fits",
    "EFOSC.2001-01-03T08:55:18.187.fits",
    "EFOSC.2001-01-03T08:56:03.899.fits",
]

for folder in zenodo_path.iterdir():
    if folder.is_dir():
        new_folder_path = raw_path / folder.name
        new_folder_path.mkdir(parents=True, exist_ok=True)

        if folder.name == "CALIB":
            collection = ccdp.ImageFileCollection(folder)
            object_files = [
                collection.files_filtered(
                    NAXIS1=IMG_DIM, NAXIS2=IMG_DIM, OBJECT=object_name
                )
                for object_name in calib_object_names
            ]

            for object_name, files in zip(calib_object_names, object_files):
                object_folder = new_folder_path / object_name
                object_folder.mkdir(parents=True, exist_ok=True)

                for file in files:
                    Path(file).rename(object_folder / file.name)

            for standard_star in standard_star_filenames:
                object_folder = new_folder_path / "STANDARD_STARS"
                object_folder.mkdir(parents=True, exist_ok=True)
                standard_star_path = folder / standard_star
                standard_star_path.rename(object_folder / standard_star_path.name)

        else:
            for file in folder.iterdir():
                if file.is_file():
                    file.rename(new_folder_path / file.name)
