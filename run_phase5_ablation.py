"""
Ablation behind the "Improved vs Phase 3 baseline particle filter" table in the README.

The improved particle filter makes two changes to the Phase 3 filter; this runs all four combinations:
    likelihood       1/(MSE+eps)  or  Gaussian exp(-SSE / (2 sigma^2))
    particle source  classifier + error samples  or  PnP pose + calibrated R
N = 1000, alpha = 0.9 every step, approach trajectory (no outage), noise 0, 2, 5 px, 5 seeds.
Prints mean position [m] / attitude [deg], each the mean over seeds of the per-run mean error.
Needs results/kf_noise_calibration.json (written by run_phase5.py).

    python run_phase5_ablation.py
"""

import json
import os

import numpy as np

import classifier
import config
import filters
import simulator
from run_phase5 import errors, stat

NOISES = [0.0, 2.0, 5.0]
SEEDS = range(5)
N_PARTICLES = 1000


class AblationPF(filters.ImprovedParticleFilter):
    """ImprovedParticleFilter with each of its two changes switchable."""

    def __init__(self, model, noise_px, R_ref, gaussian, pnp_source, seed=0):
        super().__init__(model, N_PARTICLES, filters.alpha_every_step(0.9), noise_px, R_ref, seed)
        self.gaussian = gaussian
        self.pnp_source = pnp_source

    def _draw_from_classifier(self, obs, m):
        if self.pnp_source:
            return super()._draw_from_classifier(obs, m)
        return filters.ParticleFilter._draw_from_classifier(self, obs, m)

    def update_weights(self, obs):
        if self.gaussian:
            return super().update_weights(obs)
        return filters.ParticleFilter.update_weights(self, obs)


VARIANTS = [
    ("1/(MSE+eps) + classifier (baseline design, N = 1000)", False, False),
    ("Gaussian + classifier", True, False),
    ("1/(MSE+eps) + PnP", False, True),
    ("Gaussian + PnP (improved)", True, True),
]


def main():
    with open(os.path.join(config.RESULTS_DIR, "kf_noise_calibration.json")) as f:
        calib = json.load(f)["by_noise_px"]
    model = classifier.load_model()
    times, true = simulator.approach_trajectory()

    result = {}
    for name, gaussian, pnp_source in VARIANTS:
        for noise in NOISES:
            R_ref = np.array(calib[str(noise)]["cov"])
            pos, att = [], []
            for seed in SEEDS:
                filt = AblationPF(model, noise, R_ref, gaussian, pnp_source, seed)
                est, _ = filters.run_filter(times, true, filt, noise_px=noise, seed=seed)
                p, a = errors(true, est)
                pos.append(stat(np.mean, p))
                att.append(stat(np.mean, a))
            result[(name, noise)] = (np.nanmean(pos), np.nanmean(att))
        print(f"  {name} done")

    print("\n| Likelihood / particle source | " + " | ".join(f"{n:g} px" for n in NOISES) + " |")
    print("|------|" + "------|" * len(NOISES))
    for name, _, _ in VARIANTS:
        cells = " | ".join(f"{result[(name, n)][0]:.2f} / {result[(name, n)][1]:.2f}" for n in NOISES)
        print(f"| {name} | {cells} |")


if __name__ == "__main__":
    main()
