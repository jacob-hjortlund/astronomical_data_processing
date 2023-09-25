rule master_flats:
    output:
        directory("src/data/processsed_photometry/CALIB/flats")
    cache:
        True
    script:
        "src/scripts/craete_master_flats.py"