# ------------------------------- INTERMEDIATE RESULTS ------------------------------- #

rule download_raw_data:
    input:
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-28T22:11:19.687.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:17:49.057.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-28T22:09:41.589.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-10-27T20:41:24.411.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-04T00:15:36.618.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T05:23:58.412.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T09:12:23.518.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-28T22:11:52.400.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T04:19:22.088.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T04:20:28.315.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-10-28T10:08:03.127.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:33:18.674.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-10-28T09:37:30.239.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T09:11:30.098.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:35:34.841.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:21:06.095.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:38:31.003.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:32:43.628.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:20:00.851.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:38:10.034.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T09:16:13.309.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T09:22:38.580.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:10:39.402.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T22:08:05.339.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:19:27.625.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-04T00:23:59.656.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:14:45.207.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-10-27T10:02:55.312.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T06:56:30.101.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-04T00:13:47.773.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:33:53.981.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T09:18:49.884.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:33:07.681.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T06:59:33.081.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-03T08:48:55.476.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-03T08:56:03.899.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:41:58.131.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T05:27:50.801.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T05:18:31.412.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:32:08.470.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:31:58.047.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:18:54.760.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T09:17:31.855.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T09:13:57.300.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:42:39.650.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T09:08:18.385.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T22:08:38.107.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T04:21:01.051.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T09:16:51.770.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T05:21:14.698.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:36:13.532.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:13:41.076.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:44:10.564.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T06:55:55.365.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:20:00.381.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T04:39:50.783.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T22:07:32.408.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-28T13:35:24.270.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T05:26:12.621.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-04T00:24:50.308.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T00:19:16.506.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T09:14:44.559.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:36:28.015.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T09:09:29.127.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:20:33.429.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-03T08:48:04.849.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-03T08:55:18.187.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:18:21.925.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:32:32.841.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T04:38:45.156.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T00:20:22.775.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-04T00:14:59.956.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T05:19:53.067.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:15:02.963.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:13:23.546.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T22:09:43.891.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:30:57.048.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:17:46.494.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T05:25:39.820.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T06:58:14.598.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T09:20:49.768.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T06:57:04.734.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T00:19:49.346.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T05:22:36.599.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:19:27.855.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T05:27:18.121.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T07:01:36.689.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-03T08:49:41.042.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:37:31.359.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:44:52.264.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-04T00:22:25.677.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-03T08:54:27.270.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T04:21:33.840.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T09:18:11.147.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T00:18:11.087.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T07:00:55.486.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-28T22:10:46.963.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-04T00:23:11.456.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:33:42.199.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:30:20.105.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:34:16.752.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:35:47.059.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-10-27T20:10:52.021.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T00:18:43.730.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T04:19:55.000.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T09:16:55.706.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T06:57:39.620.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-28T22:10:14.308.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:12:19.406.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:29:43.274.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T09:15:29.710.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T04:40:56.374.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T09:21:26.339.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-04T00:16:13.424.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-04T00:21:41.539.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:12:01.644.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:16:24.599.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:34:28.948.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:16:07.522.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:45:34.118.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T04:40:23.593.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T05:26:45.460.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T07:00:14.337.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2001-01-04T00:14:23.696.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-10-27T09:32:22.516.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:21:38.719.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-28T21:34:10.326.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:31:41.617.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T09:10:32.319.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T09:19:28.203.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T09:22:02.410.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:43:21.516.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T14:32:08.824.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T04:39:18.017.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T09:23:14.312.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:32:55.178.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:37:08.959.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T06:32:18.327.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-31T05:37:50.053.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T22:09:10.995.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-29T07:02:17.818.fits",
        "src/data/zenodo_archive/CALIB/EFOSC.2000-12-30T05:36:52.414.fits",
        "src/data/zenodo_archive/SCIENCE/EFOSC.2000-12-30T07:58:50.968.fits",
        "src/data/zenodo_archive/SCIENCE/EFOSC.2000-12-30T08:00:22.811.fits",
        "src/data/zenodo_archive/SCIENCE/EFOSC.2000-12-30T07:59:36.806.fits",
    output:
        directory("src/data/raw_photometry"),
    cache:
        True
    script:
        "src/scripts/intermediate_results/download_raw_data.py"

rule create_master_biases:
    input:
        "src/data/raw_photometry",
        "src/scripts/intermediate_results/processing_utils.py"
    output:
        directory("src/data/processed_photometry/calibration/bias"),
    cache:
        True
    script:
        "src/scripts/intermediate_results/create_master_biases.py"

rule create_master_dark:
    input:
        "src/data/raw_photometry",
        "src/data/processed_photometry/calibration/bias",
        "src/scripts/intermediate_results/processing_utils.py"
    output:
        directory("src/data/processed_photometry/calibration/darks"),
    cache:
        True
    script:
        "src/scripts/intermediate_results/create_master_dark.py"

rule create_master_flats:
    input:
        "src/data/raw_photometry",
        "src/data/processed_photometry/calibration/bias",
        "src/data/processed_photometry/calibration/darks",
        "src/scripts/intermediate_results/processing_utils.py"
    output:
        directory("src/data/processed_photometry/calibration/flats"),
    cache:
        True
    script:
        "src/scripts/intermediate_results/create_master_flats.py"

rule process_science_images:
    input:
        "src/data/raw_photometry",
        "src/data/processed_photometry/calibration/bias",
        "src/data/processed_photometry/calibration/darks",
        "src/data/processed_photometry/calibration/flats",
        "src/scripts/intermediate_results/processing_utils.py"
    cache:
        True
    output:
        directory("src/data/processed_photometry/science/observations"),
    script:
        "src/scripts/intermediate_results/process_science_images.py"

rule process_standard_star_images:
    input:
        "src/data/raw_photometry",
        "src/data/processed_photometry/calibration/bias",
        "src/data/processed_photometry/calibration/darks",
        "src/data/processed_photometry/calibration/flats",
        "src/scripts/intermediate_results/processing_utils.py"
    cache:
        True
    output:
        directory("src/data/processed_photometry/science/standard_stars/fits"),
    script:
        "src/scripts/intermediate_results/process_standard_star_images.py"
rule correct_astrometry:
    input:
        "src/data/processed_photometry/science/standard_stars/fits",
    cache:
        True
    output:
        directory("src/data/processed_photometry/science/standard_stars/wcs"),
    script:
        "src/scripts/intermediate_results/correct_astrometry.py"
rule bias_frames_means_stds:
    input:
        "src/data/raw_photometry"
    cache:
        True
    output:
        "src/data/processed_photometry/numbers/bias/bias_frames_means_stds.csv"
    script:
        "src/scripts/intermediate_results/bias_frames_means_stds.py"
rule standard_star_fwhm:
    input:
        "src/data/processed_photometry/science/standard_stars/fits",
        "src/scripts/intermediate_results/processing_utils.py"
    cache:
        True
    output:
        directory("src/data/processed_photometry/numbers/standard_star_fwhm")
    script:
        "src/scripts/intermediate_results/standard_star_fwhm.py"
rule standard_star_aperture_photometry:
    input:
        "src/data/processed_photometry/science/standard_stars/fits",
        "src/data/processed_photometry/numbers/standard_star_fwhm",
        "src/scripts/intermediate_results/processing_utils.py"
    cache:
        True
    output:
        directory("src/data/processed_photometry/numbers/standard_star_aperture_phot")
    script:
        "src/scripts/intermediate_results/standard_star_aperture_photometry.py"
rule standard_star_photometric_calibration:
    input:
        "src/data/processed_photometry/science/standard_stars/fits",
        "src/data/processed_photometry/numbers/standard_star_aperture_phot",
        "src/scripts/intermediate_results/processing_utils.py"
    cache:
        True
    output:
        directory("src/data/processed_photometry/numbers/standard_star_calibration")
    script:
        "src/scripts/intermediate_results/standard_star_photometric_calibration.py"

# ------------------------------- NUMBERS ------------------------------- #

rule ron_estimates:
    input:
        "src/data/processed_photometry/numbers/bias/bias_frames_means_stds.csv",
        "src/scripts/numbers/number_utils.py"
    output:
        "src/tex/output/ron_estimates.dat"
    cache:
        True
    script:
        "src/scripts/numbers/ron_estimates.py"

# ------------------------------- FIGURES ------------------------------- #

rule random_bias_frames:
    input:
        "src/data/raw_photometry",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/random_bias_frames.pdf",
    cache:
        True
    script:
        "src/scripts/figures/random_bias_frames.py"

rule bias_frame_histograms:
    input:
        "src/data/raw_photometry",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/bias_frame_histograms.pdf",
    cache:
        True
    script:
        "src/scripts/figures/bias_frame_histograms.py"
rule master_bias_frame:
    input:
        "src/data/processed_photometry/calibration/bias",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/master_bias_frame.pdf",
    cache:
        True
    script:
        "src/scripts/figures/master_bias_frame.py"
rule master_bias_stds:
    input:
        "src/data/processed_photometry/calibration/bias",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/master_bias_stds.pdf",
    cache:
        True
    script:
        "src/scripts/figures/master_bias_stds.py"
rule master_dark_frame:
    input:
        "src/data/processed_photometry/calibration/darks",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/master_dark_frame.pdf",
    cache:
        True
    script:
        "src/scripts/figures/master_dark_frame.py"
rule master_lampflat_frames:
    input:
        "src/data/processed_photometry/calibration/flats",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/master_lampflat_frames.pdf",
    cache:
        True
    script:
        "src/scripts/figures/master_lampflat_frames.py"
rule master_skyflat_frames:
    input:
        "src/data/processed_photometry/calibration/flats",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/master_skyflat_frames.pdf",
    cache:
        True
    script:
        "src/scripts/figures/master_skyflat_frames.py"
rule lampflat_calibrated_science_images:
    input:
        "src/data/processed_photometry/science/observations",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/lampflat_calibrated_science_images.pdf",
    cache:
        True
    script:
        "src/scripts/figures/lampflat_calibrated_science_images.py"
rule skyflat_calibrated_science_images:
    input:
        "src/data/processed_photometry/science/observations",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/skyflat_calibrated_science_images.pdf",
    cache:
        True
    script:
        "src/scripts/figures/skyflat_calibrated_science_images.py"

rule lampflat_calibrated_standard_star_images:
    input:
        "src/data/processed_photometry/science/standard_stars/fits",
        "src/data/processed_photometry/science/standard_stars/wcs",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/lampflat_calibrated_standard_star_images.pdf",
    cache:
        True
    script:
        "src/scripts/figures/lampflat_calibrated_standard_star_images.py"
rule skyflat_calibrated_standard_star_images:
    input:
        "src/data/processed_photometry/science/standard_stars/fits",
        "src/data/processed_photometry/science/standard_stars/wcs",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/skyflat_calibrated_standard_star_images.pdf",
    cache:
        True
    script:
        "src/scripts/figures/skyflat_calibrated_standard_star_images.py"
rule standard_star_positions:
    input:
        "src/data/processed_photometry/science/standard_stars/fits",
        "src/data/processed_photometry/science/standard_stars/wcs",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/standard_star_positions.pdf",
    cache:
        True
    script:
        "src/scripts/figures/standard_star_positions.py"
rule standard_star_calibration_posterior:
    input:
        "src/data/processed_photometry/numbers/standard_star_calibration",
        "src/scripts/figures/figure_utils.py"
    output:
        "src/tex/figures/standard_star_calibration.pdf",
    cache:
        True
    script:
        "src/scripts/figures/standard_star_photometric_calibration.py"

# ------------------------------- NUMBERS ------------------------------- #
