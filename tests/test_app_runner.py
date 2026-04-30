# Tests for the reusable desktop-app workflow

import numpy as np
import soundfile as sf

from hpss.app_runner import AppSettings, run_hpss_workflow


def test_app_runner_saves_stems_and_metrics(tmp_path) -> None:
    sr = 8000
    duration = 0.5
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    audio = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    input_path = tmp_path / "tone.wav"
    output_dir = tmp_path / "output"
    sf.write(input_path, audio, sr)

    messages: list[str] = []
    settings = AppSettings(
        input_path=input_path,
        output_dir=output_dir,
        n_fft=512,
        hop_length=128,
        harmonic_kernel=9,
        percussive_kernel=9,
        margin_h=1.5,
        margin_p=1.5,
        render_plots=False,
    )

    result = run_hpss_workflow(settings, progress=messages.append)

    assert result.sample_rate == sr
    assert result.hpss.harmonic.shape == audio.shape
    assert result.hpss.percussive.shape == audio.shape
    assert result.hpss.residual.shape == audio.shape
    assert "harmonic_energy_ratio" in result.metrics
    assert "Done." in messages

    for key in ("harmonic", "percussive", "residual"):
        assert key in result.output_paths
        assert result.output_paths[key].exists()
