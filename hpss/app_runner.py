# Reusable workflow for the HPSS desktop app

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .core import HPSSResult, HarmonicPercussiveSeparator
from .evaluation import HPSSVisualizer, compute_metrics
from .utils import load_audio, save_audio

ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class AppSettings:
    # User-configurable processing and export settings

    input_path: Path
    output_dir: Path = Path("output")
    n_fft: int = 2048
    hop_length: int = 512
    harmonic_kernel: int = 31
    percussive_kernel: int = 31
    margin_h: float = 2.0
    margin_p: float = 2.0
    power: float = 2.0
    mono: bool = False
    output_format: str = "wav"
    refine_percussive: bool = False
    render_plots: bool = True


@dataclass(slots=True)
class AppRunResult:
    # All artifacts produced by one desktop-app processing run

    settings: AppSettings
    audio: np.ndarray
    sample_rate: int
    hpss: HPSSResult
    metrics: dict[str, float]
    output_paths: dict[str, Path] = field(default_factory=dict)
    run_dir: Path | None = None

    @property
    def duration_seconds(self) -> float:
        """Duration of the source audio in seconds."""
        return float(self.audio.shape[-1] / self.sample_rate)


def run_hpss_workflow(
    settings: AppSettings,
    progress: ProgressCallback | None = None,
) -> AppRunResult:
    # Run load, separation, metrics, optional plots, and exports
    _validate_settings(settings)
    report = progress or (lambda _message: None)

    input_path = Path(settings.input_path)
    output_dir = Path(settings.output_dir)
    base_name = input_path.stem
    run_dir = output_dir / base_name
    stems_dir = run_dir / "stems"
    plots_dir = run_dir / "plots"

    for directory in (stems_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    report("Loading audio...")
    audio, sample_rate = load_audio(input_path, sr=None, mono=settings.mono)

    report("Separating harmonic, percussive, and residual stems...")
    separator = HarmonicPercussiveSeparator(
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        harmonic_kernel=settings.harmonic_kernel,
        percussive_kernel=settings.percussive_kernel,
        power=settings.power,
    )
    result = separator.separate(
        audio,
        margin_h=settings.margin_h,
        margin_p=settings.margin_p,
        refine_percussive=settings.refine_percussive,
    )

    ext = settings.output_format
    output_paths: dict[str, Path] = {
        "harmonic": stems_dir / f"{base_name}_harmonic.{ext}",
        "percussive": stems_dir / f"{base_name}_percussive.{ext}",
        "residual": stems_dir / f"{base_name}_residual.{ext}",
    }

    report("Saving stems...")
    save_audio(output_paths["harmonic"], result.harmonic, sample_rate)
    save_audio(output_paths["percussive"], result.percussive, sample_rate)
    save_audio(output_paths["residual"], result.residual, sample_rate)

    report("Computing metrics...")
    metrics = compute_metrics(audio, result.harmonic, result.percussive, residual=result.residual)

    if settings.render_plots:
        report("Generating plots...")
        visualizer = HPSSVisualizer(sample_rate, hop_length=settings.hop_length)
        spectrogram_fig = visualizer.plot_spectrograms(audio, result.harmonic, result.percussive)
        output_paths["spectrograms"] = plots_dir / f"{base_name}_spectrograms.png"
        spectrogram_fig.savefig(output_paths["spectrograms"], dpi=150, bbox_inches="tight")
        plt.close(spectrogram_fig)

        masks_fig = visualizer.plot_masks(result.harmonic_mask, result.percussive_mask)
        output_paths["masks"] = plots_dir / f"{base_name}_masks.png"
        masks_fig.savefig(output_paths["masks"], dpi=150, bbox_inches="tight")
        plt.close(masks_fig)

    report("Done.")
    return AppRunResult(
        settings=settings,
        audio=audio,
        sample_rate=sample_rate,
        hpss=result,
        metrics=metrics,
        output_paths=output_paths,
        run_dir=run_dir,
    )


def _validate_settings(settings: AppSettings) -> None:
    input_path = Path(settings.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if settings.output_format not in {"wav", "flac"}:
        raise ValueError("output_format must be 'wav' or 'flac'")
    if settings.n_fft <= 0 or settings.hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")
    if settings.harmonic_kernel <= 0 or settings.percussive_kernel <= 0:
        raise ValueError("kernel sizes must be positive")
    if settings.margin_h <= 0 or settings.margin_p <= 0:
        raise ValueError("margins must be positive")
    if settings.power <= 0:
        raise ValueError("power must be positive")
