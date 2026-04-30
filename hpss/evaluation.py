# Evaluation metrics and visualization helpers for HPSS

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from .utils import to_mono


def compute_metrics(
    original: np.ndarray,
    harmonic: np.ndarray,
    percussive: np.ndarray,
    residual: np.ndarray | None = None,
) -> dict[str, float]:
    # Compute lightweight metrics for inspecting a separation result
    original = np.asarray(original, dtype=float)
    harmonic = np.asarray(harmonic, dtype=float)
    percussive = np.asarray(percussive, dtype=float)
    if residual is not None:
        residual = np.asarray(residual, dtype=float)
        min_len = min(original.shape[-1], harmonic.shape[-1], percussive.shape[-1], residual.shape[-1])
    else:
        min_len = min(original.shape[-1], harmonic.shape[-1], percussive.shape[-1])
    original = original[..., :min_len]
    harmonic = harmonic[..., :min_len]
    percussive = percussive[..., :min_len]
    if residual is not None:
        residual = residual[..., :min_len]

    total_energy = float(np.sum(original**2)) + np.finfo(float).eps
    h_energy = float(np.sum(harmonic**2))
    p_energy = float(np.sum(percussive**2))
    if residual is None:
        residual = original - (harmonic + percussive)
    residual_energy = float(np.sum(residual**2))

    if residual.size:
        residual_mono = to_mono(residual)
        flatness_fft = min(2048, residual_mono.size)
        residual_flatness = float(
            librosa.feature.spectral_flatness(y=residual_mono, n_fft=flatness_fft)[0].mean()
        )
    else:
        residual_flatness = 0.0

    return {
        "harmonic_energy_ratio": h_energy / total_energy,
        "percussive_energy_ratio": p_energy / total_energy,
        "residual_energy_ratio": residual_energy / total_energy,
        "separation_quality_db": 10.0 * np.log10(total_energy / (residual_energy + np.finfo(float).eps)),
        "residual_spectral_flatness": residual_flatness,
    }


class HPSSVisualizer:
    # Generate report-ready plots for HPSS outputs

    def __init__(self, sr: int, hop_length: int = 512) -> None:
        self.sr = sr
        self.hop_length = hop_length

    def plot_spectrograms(
        self,
        original: np.ndarray,
        harmonic: np.ndarray,
        percussive: np.ndarray,
        figsize: tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        # Plot original, harmonic, percussive, and residual spectrograms
        original = np.asarray(original, dtype=float)
        harmonic = np.asarray(harmonic, dtype=float)
        percussive = np.asarray(percussive, dtype=float)
        min_len = min(original.shape[-1], harmonic.shape[-1], percussive.shape[-1])
        original = to_mono(original[..., :min_len])
        harmonic = to_mono(harmonic[..., :min_len])
        percussive = to_mono(percussive[..., :min_len])
        residual = original - (harmonic + percussive)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        panels = [
            ("Original Mix", original, axes[0, 0]),
            ("Harmonic Component", harmonic, axes[0, 1]),
            ("Percussive Component", percussive, axes[1, 0]),
            ("Residual", residual, axes[1, 1]),
        ]

        for title, audio, ax in panels:
            spectrogram = librosa.amplitude_to_db(
                np.abs(librosa.stft(audio, hop_length=self.hop_length)),
                ref=np.max,
            )
            image = librosa.display.specshow(
                spectrogram,
                sr=self.sr,
                hop_length=self.hop_length,
                x_axis="time",
                y_axis="log",
                ax=ax,
            )
            ax.set_title(title)
            fig.colorbar(image, ax=ax, format="%+2.0f dB")

        fig.tight_layout()
        return fig

    def plot_masks(
        self,
        mask_h: np.ndarray,
        mask_p: np.ndarray,
        figsize: tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        # Visualize harmonic and percussive soft masks
        if mask_h.ndim == 3:
            mask_h = np.mean(mask_h, axis=0)
        if mask_p.ndim == 3:
            mask_p = np.mean(mask_p, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for title, mask, ax in [
            ("Harmonic Soft Mask", mask_h, axes[0]),
            ("Percussive Soft Mask", mask_p, axes[1]),
        ]:
            image = ax.imshow(mask, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(title)
            ax.set_xlabel("Time frames")
            ax.set_ylabel("Frequency bins")
            fig.colorbar(image, ax=ax)

        fig.tight_layout()
        return fig
