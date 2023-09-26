import paths

zenodo_path = paths.data / "zenodo_archive"
raw_path = paths.data / "raw_photometry"
print(zenodo_path)
print(raw_path)

for folder in zenodo_path.iterdir():
    if folder.is_dir():
        new_folder_path = raw_path / folder.name
        new_folder_path.mkdir(parents=True, exist_ok=True)
        for file in folder.iterdir():
            if file.is_file():
                file.rename(new_folder_path / file.name)
        # folder.rmdir()
