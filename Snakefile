rule master_bias:
    input:
        "src/data/raw_photometry/CALIB"
    output:
        directory("src/data/processed_photometry/CALIB/bias"),
        "src/figures/bias_frame_stats.pdf"
    script:
        "src/scripts/create_master_biases.py"

rule master_bias_figure:
    input:
        "src/data/processed_photometry/CALIB/bias"
    output:
        "src/figures/master_bias.pdf"
    script:
        "src/scripts/master_bias_figure.py"

rule master_dark:
    input:
        "src/data/raw_photometry/CALIB",
        "src/data/processed_photometry/CALIB/bias"
    output:
        directory("src/data/processed_photometry/CALIB/darks"),
        "src/figures/master_dark.pdf"
    script:
        "src/scripts/create_master_dark.py"

rule master_flats:
    input:
        "src/data/raw_photometry/CALIB",
        "src/data/processed_photometry/CALIB/bias",
        "src/data/processed_photometry/CALIB/darks"
    output:
        directory("src/data/processed_photometry/CALIB/flats")
    # cache:
    #     True
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