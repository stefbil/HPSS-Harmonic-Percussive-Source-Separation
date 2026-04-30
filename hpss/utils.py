from pathlib import Path

import librosa
import numpy as np
import soundfile as sf



def load_audio(path: str | Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    # Load an audio file as floating-point samples
    y, sample_rate = librosa.load(path, sr=sr, mono=mono)
    return y, sample_rate


def to_mono(y: np.ndarray) -> np.ndarray:
    # Convert channel-first audio to mono for analysis and plotting
    audio = np.asarray(y)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=0)
    raise ValueError("audio must be a 1D mono or 2D channel-first array")


def normalize_audio(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    # Peak-normalize audio only when it would otherwise clip
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs > peak:
        return y / max_abs * peak
    return y


def to_soundfile_shape(y: np.ndarray) -> np.ndarray:
    # Convert channel-first audio into soundfile's frame-first layout
    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.T
    raise ValueError("audio must be a 1D mono or 2D channel-first array")


def save_audio(path: str | Path, y: np.ndarray, sr: int, normalize: bool = True) -> None:
    # Save audio to disk
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(y, dtype=np.float32)
    if normalize:
        audio = normalize_audio(audio)

    sf.write(output_path, to_soundfile_shape(audio), sr)
