import paths
import pandas as pd
import number_utils as nu


frame_names = ["frame_1", "frame_2", "diff"]
stats_path = (
    paths.data / "processed_photometry" / "calibration" / "bias" / "bias_stats.csv"
)
stats_df = pd.read_csv(stats_path, index_col=0)

for frame_name in frame_names:
    frame_std = stats_df.loc[frame_name, "std"]
    nu.save_variable_to_latex(
        variable=frame_std,
        sigfigs=2,
        variable_name=f"{frame_name}",
        filename="ron_estimates.dat",
        path=paths.output,
    )
