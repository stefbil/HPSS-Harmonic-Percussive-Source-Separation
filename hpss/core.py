# Core HPSS algorithm using median-filter spectrogram decomposition

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import median_filter


@dataclass(slots=True)
class HPSSResult:
    # Separated audio and masks produced by HPSS

    harmonic: np.ndarray
    percussive: np.ndarray
    residual: np.ndarray
    harmonic_mask: np.ndarray
    percussive_mask: np.ndarray


class HarmonicPercussiveSeparator:
    # Separate audio into harmonic, percussive, and residual components.

    # The implementation follows the median-filtering method described by
    # Fitzgerald, horizontal spectrogram structures are treated as
    # harmonic, while vertical structures are treated as percussive.
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int | None = None,
        harmonic_kernel: int = 31,
        percussive_kernel: int = 31,
        power: float = 2.0,
    ) -> None:
        if n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if harmonic_kernel <= 0 or percussive_kernel <= 0:
            raise ValueError("median filter kernels must be positive")
        if power <= 0:
            raise ValueError("power must be positive")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.harmonic_kernel = self._ensure_odd(harmonic_kernel)
        self.percussive_kernel = self._ensure_odd(percussive_kernel)
        self.power = power

    def forward(
        self,
        y: np.ndarray,
        margin_h: float = 1.0,
        margin_p: float = 1.0,
        refine_percussive: bool = False,
    ) -> HPSSResult:
        # Run HPSS and return a named result object
        return self.separate(
            y,
            margin_h=margin_h,
            margin_p=margin_p,
            refine_percussive=refine_percussive,
        )

    def separate(
        self,
        y: np.ndarray,
        margin_h: float = 1.0,
        margin_p: float = 1.0,
        refine_percussive: bool = False,
    ) -> HPSSResult:
        # Separate mono or channel-first stereo audio
        if margin_h <= 0 or margin_p <= 0:
            raise ValueError("margins must be positive")

        audio = np.asarray(y, dtype=float)
        if audio.size == 0:
            raise ValueError("audio input is empty")
        if audio.ndim == 1:
            return self._separate_mono(
                audio,
                margin_h=margin_h,
                margin_p=margin_p,
                refine_percussive=refine_percussive,
            )
        if audio.ndim == 2:
            channel_results = [
                self._separate_mono(
                    channel,
                    margin_h=margin_h,
                    margin_p=margin_p,
                    refine_percussive=refine_percussive,
                )
                for channel in audio
            ]
            return HPSSResult(
                harmonic=np.stack([result.harmonic for result in channel_results]),
                percussive=np.stack([result.percussive for result in channel_results]),
                residual=np.stack([result.residual for result in channel_results]),
                harmonic_mask=np.stack([result.harmonic_mask for result in channel_results]),
                percussive_mask=np.stack([result.percussive_mask for result in channel_results]),
            )

        raise ValueError("HPSS expects a 1D mono or 2D channel-first audio array")

    def _separate_mono(
        self,
        audio: np.ndarray,
        margin_h: float,
        margin_p: float,
        refine_percussive: bool = False,
    ) -> HPSSResult:
        # Separate one mono channe
        D = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
        )
        magnitude = np.abs(D)

        harmonic_enhanced = median_filter(
            magnitude,
            size=(1, self.harmonic_kernel),
            mode="reflect",
        )
        percussive_enhanced = median_filter(
            magnitude,
            size=(self.percussive_kernel, 1),
            mode="reflect",
        )

        mask_h = self._compute_mask(harmonic_enhanced, percussive_enhanced, margin_h)
        mask_p = self._compute_mask(percussive_enhanced, harmonic_enhanced, margin_p)

        D_h = D * mask_h
        D_p = D * mask_p

        y_h = librosa.istft(
            D_h,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=audio.shape[-1],
        )
        y_p = librosa.istft(
            D_p,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=audio.shape[-1],
        )
        if refine_percussive:
            y_p = librosa.griffinlim(
                np.abs(D_p),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_iter=16,
                length=audio.shape[-1],
            )
        residual = audio - (y_h + y_p)

        return HPSSResult(y_h, y_p, residual, mask_h, mask_p)

    def _compute_mask(self, target: np.ndarray, other: np.ndarray, margin: float) -> np.ndarray:
        # Compute a soft mask for one enhanced spectrogram against another
        eps = np.finfo(float).eps
        target_power = np.power(target, self.power)
        other_power = np.power(margin * other, self.power)
        return target_power / (target_power + other_power + eps)

    @staticmethod
    def _ensure_odd(value: int) -> int:
        # Return an odd median-filter kernel size
        return value if value % 2 == 1 else value + 1
