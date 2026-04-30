#!/usr/bin/env python3
#CLI for HPSS

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from hpss.core import HarmonicPercussiveSeparator
from hpss.evaluation import HPSSVisualizer, compute_metrics
from hpss.utils import load_audio, save_audio

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Separate a music mix into harmonic, percussive, and residual stems."
    )
    parser.add_argument("input", type=Path, help="Input audio file path")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output"))

    parser.add_argument("--n-fft", type=int, default=2048, help="FFT size")
    parser.add_argument("--hop-length", type=int, default=512, help="STFT hop length")
    parser.add_argument("--h-kernel", type=int, default=31, help="Harmonic median filter width")
    parser.add_argument("--p-kernel", type=int, default=31, help="Percussive median filter width")
    parser.add_argument("--margin-h", type=float, default=2.0, help="Harmonic mask margin")
    parser.add_argument("--margin-p", type=float, default=2.0, help="Percussive mask margin")
    parser.add_argument("--power", type=float, default=2.0, help="Soft-mask exponent")

    parser.add_argument("--format", choices=("wav", "flac"), default="wav", help="Output audio format")
    parser.add_argument("--plot", action="store_true", help="Save spectrogram and mask figures")
    parser.add_argument("--mono", action="store_true", help="Downmix input to mono before separation")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    # Run the HPSS command-line workflow
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    base_name = args.input.stem
    run_dir = args.output_dir / base_name
    stems_dir = run_dir / "stems"
    plots_dir = run_dir / "plots"

    for directory in (stems_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger.info("Loading audio: %s", args.input)
    y, sr = load_audio(args.input, sr=None, mono=args.mono)
    channels = 1 if y.ndim == 1 else y.shape[0]
    logger.info("Sample rate: %s Hz, channels: %s, duration: %.2f s", sr, channels, y.shape[-1] / sr)
    logger.info("Output folder: %s", run_dir)

    separator = HarmonicPercussiveSeparator(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        harmonic_kernel=args.h_kernel,
        percussive_kernel=args.p_kernel,
        power=args.power,
    )

    logger.info("Running HPSS...")
    result = separator.forward(
        y,
        margin_h=args.margin_h,
        margin_p=args.margin_p,
    )

    h_path = stems_dir / f"{base_name}_harmonic.{args.format}"
    p_path = stems_dir / f"{base_name}_percussive.{args.format}"
    r_path = stems_dir / f"{base_name}_residual.{args.format}"
    save_audio(h_path, result.harmonic, sr)
    save_audio(p_path, result.percussive, sr)
    save_audio(r_path, result.residual, sr)
    logger.info("Saved stem: %s", h_path)
    logger.info("Saved stem: %s", p_path)
    logger.info("Saved stem: %s", r_path)

    logger.info("")
    logger.info("Separation metrics:")
    for key, value in compute_metrics(y, result.harmonic, result.percussive, residual=result.residual).items():
        logger.info("  %s: %.4f", key, value)

    if args.plot:
        logger.info("")
        logger.info("Generating plots...")
        visualizer = HPSSVisualizer(sr, hop_length=args.hop_length)

        spectrogram_fig = visualizer.plot_spectrograms(y, result.harmonic, result.percussive)
        spectrogram_path = plots_dir / f"{base_name}_spectrograms.png"
        spectrogram_fig.savefig(spectrogram_path, dpi=150, bbox_inches="tight")
        plt.close(spectrogram_fig)
        logger.info("Saved plot: %s", spectrogram_path)

        masks_fig = visualizer.plot_masks(result.harmonic_mask, result.percussive_mask)
        masks_path = plots_dir / f"{base_name}_masks.png"
        masks_fig.savefig(masks_path, dpi=150, bbox_inches="tight")
        plt.close(masks_fig)
        logger.info("Saved plot: %s", masks_path)

    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
