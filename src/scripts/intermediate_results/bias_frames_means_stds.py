import paths
import numpy as np
import ccdproc as ccdp
import processing_utils as utils

raw_bias_path = paths.data / "raw_photometry" / "CALIB" / "BIAS"
bias_collection = ccdp.ImageFileCollection(location=raw_bias_path)
biases = list(bias_collection.ccds(ccd_kwargs={"unit": "adu"}))

frame_1 = biases[0]
frame_2 = biases[-1]
frame_diff = frame_1.subtract(frame_2)
frames = [frame_1, frame_2, frame_diff]
frame_names = ["frame_1", "frame_2", "diff"]
output_array = np.zeros((len(frames), 2))
output_path = paths.data / "processed_photometry" / "numbers" / "bias"

for i, (frame, frame_name) in enumerate(zip(frames, frame_names)):
    frame_mean = frame.data.mean()
    frame_std = frame.data.std()
    if i == 2:
        frame_std /= np.sqrt(2)
    xlim = (frame_mean - 5 * frame_std, frame_mean + 5 * frame_std)

    idx_clipped = (frame.data > xlim[0]) & (frame.data < xlim[1])
    frame_data_clipped = frame.data[idx_clipped]
    frame_mean = frame_data_clipped.mean()
    frame_std = frame_data_clipped.std()
    if i == 2:
        frame_std /= np.sqrt(2)

    output_array[i, 0] = frame_mean
    output_array[i, 1] = frame_std

utils.save_array_to_csv(
    array=output_array,
    column_names=["mean", "std"],
    index_names=frame_names,
    filename="bias_frames_means_stds.csv",
    path=output_path,
    overwrite=True,
)
