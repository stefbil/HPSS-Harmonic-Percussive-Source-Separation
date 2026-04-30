# Synthetic demo: C-major chord plus drum-like impulses

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hpss.core import HarmonicPercussiveSeparator
from hpss.evaluation import HPSSVisualizer, compute_metrics
from hpss.utils import save_audio


def make_synthetic_mix(sr: int = 22050, duration: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Create a controlled harmonic/percussive test signal
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)

    chord = 0.25 * (
        np.sin(2 * np.pi * 261.63 * t)
        + np.sin(2 * np.pi * 329.63 * t)
        + np.sin(2 * np.pi * 392.00 * t)
    )

    drums = np.zeros_like(t)
    for beat_time in np.arange(0.2, duration, 0.5):
        start = int(beat_time * sr)
        end = min(start + 350, len(drums))
        decay = np.exp(-np.linspace(0.0, 6.0, end - start))
        drums[start:end] += 0.8 * decay

    mix = chord + drums
    return mix, chord, drums


def main() -> None:
    sr = 22050
    run_dir = Path("output") / "demo"
    references_dir = run_dir / "references"
    stems_dir = run_dir / "stems"
    plots_dir = run_dir / "plots"

    for directory in (references_dir, stems_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mix, chord, drums = make_synthetic_mix(sr=sr)
    save_audio(references_dir / "demo_original_mix.wav", mix, sr)
    save_audio(references_dir / "demo_reference_harmonic.wav", chord, sr)
    save_audio(references_dir / "demo_reference_percussive.wav", drums, sr)

    separator = HarmonicPercussiveSeparator(harmonic_kernel=17, percussive_kernel=17)
    result = separator.forward(mix)

    save_audio(stems_dir / "demo_harmonic.wav", result.harmonic, sr)
    save_audio(stems_dir / "demo_percussive.wav", result.percussive, sr)
    save_audio(stems_dir / "demo_residual.wav", result.residual, sr)

    print("Synthetic demo metrics:")
    for key, value in compute_metrics(mix, result.harmonic, result.percussive, residual=result.residual).items():
        print(f"  {key}: {value:.4f}")

    visualizer = HPSSVisualizer(sr)
    spectrogram_fig = visualizer.plot_spectrograms(mix, result.harmonic, result.percussive)
    spectrogram_fig.savefig(plots_dir / "demo_spectrograms.png", dpi=150, bbox_inches="tight")
    plt.close(spectrogram_fig)

    masks_fig = visualizer.plot_masks(result.harmonic_mask, result.percussive_mask)
    masks_fig.savefig(plots_dir / "demo_masks.png", dpi=150, bbox_inches="tight")
    plt.close(masks_fig)


if __name__ == "__main__":
    main()
