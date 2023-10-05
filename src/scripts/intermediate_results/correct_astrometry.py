import time
import paths
import processing_utils as utils

ASTROMETRY_API_KEY = None  # Use environment variable instead
SLEEP_TIME = 1  # seconds
SCALE_LOWER = 0.1  # arcsec / pixel lower limit
SCALE_UPPER = 0.5  # arcsec / pixel upper limit
SCALE_UNIT = "arcsecperpix"

upload_kwargs = {
    "scale_lower": SCALE_LOWER,
    "scale_upper": SCALE_UPPER,
    "scale_units": SCALE_UNIT,
}

fits_directory = (
    paths.data / "processed_photometry" / "science" / "standard_stars"
)  # Path to directory containing fits files to correct
save_directory = (
    paths.data / "processed_photometry" / "science" / "standard_stars"
)  # Path to directory to save corrected fits files

client = utils.AstrometryClient()
client.login(ASTROMETRY_API_KEY)

for fits_file_path in fits_directory.glob("*.fits"):
    upload_response = client.upload(fits_file_path, **upload_kwargs)
    submission_id = upload_response["subid"]

    print(f"\n{fits_file_path.name} uploaded")
    print(f"Submission ID: {submission_id}")

    # Get job ID for submission
    while True:
        submission_response = client.sub_status(submission_id, justdict=True)
        job_list = submission_response.get("jobs", [None])
        if len(job_list) > 0:
            if job_list[0] is not None:
                job_id = job_list[0]
                print(f"Job ID: {job_id}")
                break
        else:
            time.sleep(SLEEP_TIME)

    # Get job status
    while True:
        job_status_response = client.job_status(job_id, justdict=True)
        job_status = job_status_response.get("status", None)
        if job_status == "success":
            print(f"Astrometry solving successful")
            break
        elif job_status == "failure":
            print(f"Astrometry solving failed")
            break
        time.sleep(SLEEP_TIME)

    # Get job results
    output_fits_file = client.get_output_fits_file(job_id)
    save_path = save_directory / fits_file_path.name
    with save_path.open("wb") as f:
        f.write(output_fits_file.read())
        print(f"Astrometry results saved to {save_path}")
