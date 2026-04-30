# Unit tests for the HPSS implementation

import numpy as np
import soundfile as sf

from hpss.core import HarmonicPercussiveSeparator
from hpss.evaluation import compute_metrics
from hpss.utils import save_audio


def test_separator_instantiation_uses_odd_kernels() -> None:
    separator = HarmonicPercussiveSeparator(harmonic_kernel=16, percussive_kernel=18)

    assert separator.n_fft == 2048
    assert separator.hop_length == 512
    assert separator.harmonic_kernel == 17
    assert separator.percussive_kernel == 19


def test_pure_sine_wave_is_mostly_harmonic() -> None:
    sr = 22050
    duration = 1.5
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440.0 * t)

    separator = HarmonicPercussiveSeparator()
    result = separator.forward(y)

    h_energy = np.sum(result.harmonic**2)
    p_energy = np.sum(result.percussive**2)

    assert h_energy > 5.0 * p_energy


def test_impulse_train_is_mostly_percussive() -> None:
    sr = 22050
    duration = 1.5
    y = np.zeros(int(sr * duration))
    y[:: sr // 8] = 1.0

    separator = HarmonicPercussiveSeparator()
    result = separator.forward(y)

    h_energy = np.sum(result.harmonic**2)
    p_energy = np.sum(result.percussive**2)

    assert p_energy > h_energy


def test_outputs_match_input_length_and_masks_are_bounded() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(scale=0.1, size=22050)

    separator = HarmonicPercussiveSeparator()
    result = separator.forward(y)

    assert len(result.harmonic) == len(y)
    assert len(result.percussive) == len(y)
    assert len(result.residual) == len(y)
    np.testing.assert_allclose(result.harmonic + result.percussive + result.residual, y)
    assert np.all((0.0 <= result.harmonic_mask) & (result.harmonic_mask <= 1.0))
    assert np.all((0.0 <= result.percussive_mask) & (result.percussive_mask <= 1.0))


def test_stereo_input_preserves_channel_layout() -> None:
    sr = 22050
    duration = 1.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    stereo = np.vstack(
        [
            np.sin(2 * np.pi * 440.0 * t),
            np.sin(2 * np.pi * 660.0 * t),
        ]
    )

    separator = HarmonicPercussiveSeparator(n_fft=1024, hop_length=256)
    result = separator.forward(stereo)

    assert result.harmonic.shape == stereo.shape
    assert result.percussive.shape == stereo.shape
    assert result.residual.shape == stereo.shape
    assert result.harmonic_mask.shape[0] == 2
    assert result.percussive_mask.shape[0] == 2


def test_save_audio_writes_stereo_file(tmp_path) -> None:
    sr = 22050
    stereo = np.vstack([np.linspace(-0.2, 0.2, sr), np.linspace(0.2, -0.2, sr)])
    output_path = tmp_path / "stereo.wav"

    save_audio(output_path, stereo, sr)
    info = sf.info(output_path)

    assert info.channels == 2
    assert info.frames == sr


def test_compute_metrics_reports_low_residual_for_exact_sum() -> None:
    harmonic = np.array([0.2, 0.3, 0.2, 0.1])
    percussive = np.array([0.5, -0.1, 0.0, 0.2])
    original = harmonic + percussive

    metrics = compute_metrics(original, harmonic, percussive)

    assert metrics["residual_energy_ratio"] < 1e-12
