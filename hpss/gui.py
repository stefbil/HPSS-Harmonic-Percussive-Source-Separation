# Desktop GUI for the HPSS app

from __future__ import annotations

import queue
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .app_runner import AppRunResult, AppSettings, run_hpss_workflow
from .evaluation import HPSSVisualizer
from .utils import to_soundfile_shape


class HPSSApp(ctk.CTk):
    # Main desktop application window

    def __init__(self) -> None:
        super().__init__()
        self.title("HPSS Studio")
        self.geometry("1220x820")
        self.minsize(1060, 720)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.input_path = ctk.StringVar()
        self.output_dir = ctk.StringVar(value=str(Path("output").resolve()))
        self.status_text = ctk.StringVar(value="Choose an audio file to begin.")
        self.result: AppRunResult | None = None
        self.worker: threading.Thread | None = None
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.plot_canvas: FigureCanvasTkAgg | None = None
        self.current_figure: plt.Figure | None = None
        self.stem_buttons: dict[str, ctk.CTkButton] = {}

        self._create_variables()
        self._build_layout()
        self.after(100, self._poll_messages)

    def _create_variables(self) -> None:
        self.n_fft = ctk.IntVar(value=2048)
        self.hop_length = ctk.IntVar(value=512)
        self.h_kernel = ctk.IntVar(value=31)
        self.p_kernel = ctk.IntVar(value=31)
        self.margin_h = ctk.DoubleVar(value=2.0)
        self.margin_p = ctk.DoubleVar(value=2.0)
        self.power = ctk.DoubleVar(value=2.0)
        self.output_format = ctk.StringVar(value="wav")
        self.mono = ctk.BooleanVar(value=False)
        self.refine_percussive = ctk.BooleanVar(value=False)
        self.render_plots = ctk.BooleanVar(value=True)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkScrollableFrame(self, width=360, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        content = ctk.CTkFrame(self, corner_radius=0)
        content.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(2, weight=1)

        self._build_sidebar(sidebar)
        self._build_content(content)

    def _build_sidebar(self, parent: ctk.CTkScrollableFrame) -> None:
        ctk.CTkLabel(parent, text="HPSS Studio", font=ctk.CTkFont(size=28, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=20, pady=(20, 2)
        )
        ctk.CTkLabel(
            parent,
            text="Separate music into harmonic, percussive, and residual stems.",
            wraplength=300,
            justify="left",
        ).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 18))

        self._path_picker(parent, 2, "Input audio", self.input_path, self._choose_input)
        self._path_picker(parent, 3, "Output folder", self.output_dir, self._choose_output_dir)

        controls = ctk.CTkFrame(parent)
        controls.grid(row=4, column=0, sticky="ew", padx=16, pady=10)
        self._configure_slider_columns(controls)
        ctk.CTkLabel(controls, text="Separation", font=ctk.CTkFont(size=17, weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 6)
        )
        self._number_entry(controls, 1, "FFT size", self.n_fft)
        self._number_entry(controls, 2, "Hop length", self.hop_length)
        self._number_entry(controls, 3, "H kernel", self.h_kernel)
        self._number_entry(controls, 4, "P kernel", self.p_kernel)
        self._number_entry(controls, 5, "Power", self.power)
        self._slider(controls, 6, "Harmonic margin", self.margin_h, 0.5, 5.0)
        self._slider(controls, 7, "Percussive margin", self.margin_p, 0.5, 5.0)

        output = ctk.CTkFrame(parent)
        output.grid(row=5, column=0, sticky="ew", padx=16, pady=10)
        self._configure_slider_columns(output)
        ctk.CTkLabel(output, text="Output", font=ctk.CTkFont(size=17, weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 6)
        )
        ctk.CTkOptionMenu(output, values=["wav", "flac"], variable=self.output_format).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=6
        )
        ctk.CTkCheckBox(output, text="Downmix to mono", variable=self.mono).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=12, pady=6
        )
        ctk.CTkCheckBox(output, text="Refine percussive phase", variable=self.refine_percussive).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=12, pady=6
        )
        ctk.CTkCheckBox(output, text="Save plots", variable=self.render_plots).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=12, pady=(6, 12)
        )

        self.run_button = ctk.CTkButton(parent, text="Run Separation", height=44, command=self._start_processing)
        self.run_button.grid(row=6, column=0, sticky="ew", padx=16, pady=(14, 8))
        ctk.CTkButton(parent, text="Stop Playback", command=self._stop_playback).grid(
            row=7, column=0, sticky="ew", padx=16, pady=(0, 20)
        )

    def _build_content(self, parent: ctk.CTkFrame) -> None:
        status = ctk.CTkFrame(parent)
        status.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        status.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(status, textvariable=self.status_text, anchor="w").grid(
            row=0, column=0, sticky="ew", padx=14, pady=12
        )

        self.summary_box = ctk.CTkTextbox(parent, height=120)
        self.summary_box.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        self.summary_box.insert("1.0", "Results will appear here after processing.")
        self.summary_box.configure(state="disabled")

        self.tabs = ctk.CTkTabview(parent)
        self.tabs.grid(row=2, column=0, sticky="nsew", padx=16, pady=(8, 16))
        self.tabs.add("Preview")
        self.tabs.add("Plots")
        self.tabs.add("Files")

        self._build_preview_tab(self.tabs.tab("Preview"))
        self._build_plots_tab(self.tabs.tab("Plots"))
        self._build_files_tab(self.tabs.tab("Files"))

    def _build_preview_tab(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure((0, 1), weight=1)
        stems = [
            ("Original", "original"),
            ("Harmonic", "harmonic"),
            ("Percussive", "percussive"),
            ("Residual", "residual"),
        ]
        for index, (label, key) in enumerate(stems):
            button = ctk.CTkButton(
                parent,
                text=f"Play {label}",
                state="disabled",
                command=lambda stem=key: self._play(stem),
            )
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=12, pady=10)
            self.stem_buttons[key] = button

    def _build_plots_tab(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)
        buttons = ctk.CTkFrame(parent)
        buttons.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        buttons.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(buttons, text="Show spectrograms", command=self._show_spectrograms).grid(
            row=0, column=0, sticky="ew", padx=6, pady=6
        )
        ctk.CTkButton(buttons, text="Show masks", command=self._show_masks).grid(
            row=0, column=1, sticky="ew", padx=6, pady=6
        )
        self.plot_frame = ctk.CTkFrame(parent)
        self.plot_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        ctk.CTkLabel(self.plot_frame, text="Run separation, then choose a plot.").grid(
            row=0, column=0, padx=20, pady=20
        )

    def _build_files_tab(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self.files_box = ctk.CTkTextbox(parent)
        self.files_box.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.files_box.insert("1.0", "Saved files will appear here.")
        self.files_box.configure(state="disabled")

    def _path_picker(
        self,
        parent: ctk.CTkScrollableFrame,
        row: int,
        label: str,
        variable: ctk.StringVar,
        command: object,
    ) -> None:
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=8)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 2)
        )
        ctk.CTkEntry(frame, textvariable=variable).grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        ctk.CTkButton(frame, text="Browse", command=command).grid(row=1, column=1, padx=(0, 12), pady=(0, 8))

    def _configure_slider_columns(self, frame: ctk.CTkFrame) -> None:
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=0)

    def _number_entry(self, parent: ctk.CTkFrame, row: int, label: str, variable: object) -> None:
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, sticky="w", padx=12, pady=6)
        ctk.CTkEntry(parent, textvariable=variable, width=110).grid(row=row, column=1, sticky="e", padx=12, pady=6)

    def _slider(
        self,
        parent: ctk.CTkFrame,
        row: int,
        label: str,
        variable: ctk.DoubleVar,
        from_: float,
        to: float,
    ) -> None:
        value_label = ctk.CTkLabel(parent, text=f"{variable.get():.2f}", width=44)

        def update(value: str) -> None:
            numeric = float(value)
            variable.set(numeric)
            value_label.configure(text=f"{numeric:.2f}")

        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, sticky="w", padx=12, pady=6)
        slider = ctk.CTkSlider(parent, from_=from_, to=to, variable=variable, command=update)
        slider.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=6)
        value_label.grid(row=row, column=2, sticky="e", padx=(0, 12), pady=6)

    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[
                ("Audio files", "*.wav *.flac *.mp3 *.ogg *.aiff *.aif"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_path.set(path)

    def _choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_dir.set(path)

    def _start_processing(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            settings = self._settings_from_ui()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.run_button.configure(state="disabled", text="Processing...")
        self.status_text.set("Starting separation...")
        self.result = None
        for button in self.stem_buttons.values():
            button.configure(state="disabled")
        self._set_text(self.summary_box, "Processing. This may take a moment for longer tracks.")
        self._set_text(self.files_box, "")

        self.worker = threading.Thread(target=self._run_worker, args=(settings,), daemon=True)
        self.worker.start()

    def _settings_from_ui(self) -> AppSettings:
        input_path = Path(self.input_path.get())
        if not input_path.exists():
            raise FileNotFoundError("Choose a valid input audio file.")
        return AppSettings(
            input_path=input_path,
            output_dir=Path(self.output_dir.get()),
            n_fft=int(self.n_fft.get()),
            hop_length=int(self.hop_length.get()),
            harmonic_kernel=int(self.h_kernel.get()),
            percussive_kernel=int(self.p_kernel.get()),
            margin_h=float(self.margin_h.get()),
            margin_p=float(self.margin_p.get()),
            power=float(self.power.get()),
            mono=bool(self.mono.get()),
            output_format=self.output_format.get(),
            refine_percussive=bool(self.refine_percussive.get()),
            render_plots=bool(self.render_plots.get()),
        )

    def _run_worker(self, settings: AppSettings) -> None:
        try:
            result = run_hpss_workflow(settings, progress=lambda message: self.messages.put(("status", message)))
        except Exception as exc:
            self.messages.put(("error", exc))
        else:
            self.messages.put(("result", result))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "status":
                    self.status_text.set(str(payload))
                elif kind == "error":
                    self.run_button.configure(state="normal", text="Run Separation")
                    self.status_text.set("Processing failed.")
                    messagebox.showerror("Processing failed", str(payload))
                elif kind == "result":
                    self._handle_result(payload)  # type: ignore[arg-type]
        except queue.Empty:
            pass
        self.after(100, self._poll_messages)

    def _handle_result(self, result: AppRunResult) -> None:
        self.result = result
        self.run_button.configure(state="normal", text="Run Separation")
        self.status_text.set(f"Done. Output saved to {result.run_dir}")
        available = {
            "original",
            "harmonic",
            "percussive",
            "residual",
        }
        for key, button in self.stem_buttons.items():
            button.configure(state="normal" if key in available else "disabled")
        self._update_summary(result)
        self._update_files(result)
        if result.settings.render_plots:
            self._show_spectrograms()

    def _update_summary(self, result: AppRunResult) -> None:
        channels = 1 if result.audio.ndim == 1 else result.audio.shape[0]
        lines = [
            f"Input: {result.settings.input_path}",
            f"Sample rate: {result.sample_rate} Hz | Channels: {channels} | Duration: {result.duration_seconds:.2f} s",
            "",
            "Metrics:",
        ]
        lines.extend(f"  {key}: {value:.4f}" for key, value in result.metrics.items())
        self._set_text(self.summary_box, "\n".join(lines))

    def _update_files(self, result: AppRunResult) -> None:
        lines = [f"{name}: {path}" for name, path in sorted(result.output_paths.items())]
        self._set_text(self.files_box, "\n".join(lines))

    def _set_text(self, widget: ctk.CTkTextbox, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _play(self, stem: str) -> None:
        if self.result is None:
            messagebox.showinfo("No audio yet", "Run separation before previewing stems.")
            return
        audio = self._audio_for_stem(stem)
        if audio is None:
            messagebox.showinfo("Unavailable", f"No {stem.replace('_', ' ')} audio was rendered.")
            return
        sd.stop()
        sd.play(to_soundfile_shape(np.asarray(audio, dtype=np.float32)), self.result.sample_rate)
        self.status_text.set(f"Playing {stem.replace('_', ' ')}.")

    def _audio_for_stem(self, stem: str) -> np.ndarray | None:
        if self.result is None:
            return None
        mapping = {
            "original": self.result.audio,
            "harmonic": self.result.hpss.harmonic,
            "percussive": self.result.hpss.percussive,
            "residual": self.result.hpss.residual,
        }
        return mapping.get(stem)

    def _stop_playback(self) -> None:
        sd.stop()
        self.status_text.set("Playback stopped.")

    def _show_spectrograms(self) -> None:
        if self.result is None:
            return
        visualizer = HPSSVisualizer(self.result.sample_rate, hop_length=self.result.settings.hop_length)
        fig = visualizer.plot_spectrograms(
            self.result.audio,
            self.result.hpss.harmonic,
            self.result.hpss.percussive,
            figsize=(10, 7),
        )
        self._render_figure(fig)

    def _show_masks(self) -> None:
        if self.result is None:
            return
        visualizer = HPSSVisualizer(self.result.sample_rate, hop_length=self.result.settings.hop_length)
        fig = visualizer.plot_masks(
            self.result.hpss.harmonic_mask,
            self.result.hpss.percussive_mask,
            figsize=(10, 5),
        )
        self._render_figure(fig)

    def _render_figure(self, fig: plt.Figure) -> None:
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        if self.current_figure is not None:
            plt.close(self.current_figure)
        self.current_figure = fig
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")


def main() -> None:
    # Launch the desktop app
    app = HPSSApp()
    app.mainloop()
