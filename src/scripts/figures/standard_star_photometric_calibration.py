import paths
import corner
import emcee as em
import numpy as np
import matplotlib.pyplot as plt

chain_path = (
    paths.data / "processed_photometry" / "numbers" / "standard_star_aperture_phot"
)
chain_path.mkdir(parents=True, exist_ok=True)
reader = em.backends.HDFBackend(chain_path / "chains.h5")

tau = np.max(reader.get_autocorr_time())
burnin = int(5 * tau)
thin = int(0.5 * tau)
flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)

percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
labels = [
    r"$c_{B2}$",
    r"$c_{V2}$",
    r"$c_{R2}$",
    r"$c_{B3}$",
    r"$c_{V3}$",
    r"$c_{R3}$",
]

print("\nMCMC:")
for i, label in enumerate(labels):
    print(
        f"{label} = {percentiles[1, i]:.3f} + {percentiles[2, i] - percentiles[1, i]:.3f} - {percentiles[1, i] - percentiles[0, i]:.3f}"
    )

labels = ["$c_{B2}$", "$c_{V2}$", "$c_{R2}$", "$c_{B3}$", "$c_{V3}$", "$c_{R3}$"]
fig = corner.corner(
    flat_samples,
    labels=labels,
    truths=np.percentile(flat_samples, 50, axis=0),
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    bins=50,
    smooth=1.0,
    title_fmt=".3f",
    title_kwargs={"fontsize": 16},
    label_kwargs={"fontsize": 16},
)

fig.savefig(
    paths.data / "processed_photometry" / "figures" / "standard_star_calibration.pdf"
)
