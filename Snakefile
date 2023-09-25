rule master_flats:
    output:
        directory("src/data/processed_photometry/CALIB/flats")
    cache:
        True
    script:
        "src/scripts/create_master_flats.py"

rule master_lampflat:
    input:
        "src/data/processed_photometry/CALIB/flats"
    output:
        "src/figures/master_lampflat.pdf"
    script:
        "src/scripts/master_lampflats_figure.py"

rule master_skyflat:
    input:
        "src/data/processed_photometry/CALIB/flats"
    output:
        "src/figures/master_skyflat.pdf"
    script:
        "src/scripts/master_skyflats_figure.py"