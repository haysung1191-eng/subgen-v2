from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from queue import Empty, Queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from .cli import KO_SYNC_FINAL_PRESET, KO_SYNC_PRESET, KO_SYNC_TIGHT_PRESET, MEDIA_EXTS
    from .config import AlignmentConfig, PipelineConfig, RuntimeConfig, TimingConfig, TranscriptionConfig, VADConfig
    from .errors import SubgenError
    from .runner import run_batch_with_runtime, run_single_with_runtime
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from subgen.cli import KO_SYNC_FINAL_PRESET, KO_SYNC_PRESET, KO_SYNC_TIGHT_PRESET, MEDIA_EXTS  # type: ignore
    from subgen.config import AlignmentConfig, PipelineConfig, RuntimeConfig, TimingConfig, TranscriptionConfig, VADConfig  # type: ignore
    from subgen.errors import SubgenError  # type: ignore
    from subgen.runner import run_batch_with_runtime, run_single_with_runtime  # type: ignore


PRESETS = {
    "none": {},
    "ko-sync": KO_SYNC_PRESET,
    "ko-sync-tight": KO_SYNC_TIGHT_PRESET,
    "ko-sync-final": KO_SYNC_FINAL_PRESET,
}


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Subgen GUI")
        self.geometry("980x760")

        self.queue: Queue[tuple[str, str]] = Queue()
        self.worker: threading.Thread | None = None

        self.mode_var = tk.StringVar(value="single")
        self.preset_var = tk.StringVar(value="ko-sync-final")
        self.device_var = tk.StringVar(value="cuda")
        self.model_var = tk.StringVar(value="large-v2")
        self.show_advanced_var = tk.BooleanVar(value=False)

        self.single_input_var = tk.StringVar()
        self.single_output_var = tk.StringVar()
        self.batch_input_var = tk.StringVar()
        self.batch_output_var = tk.StringVar()

        self.alignment_var = tk.StringVar(value="on")
        self.align_device_var = tk.StringVar(value="cuda")
        self.language_var = tk.StringVar(value="ko")
        self.shift_var = tk.IntVar(value=0)
        self.vad_threshold_var = tk.DoubleVar(value=0.55)
        self.vad_min_speech_var = tk.IntVar(value=260)
        self.vad_min_silence_var = tk.IntVar(value=220)
        self.vad_pad_var = tk.IntVar(value=50)
        self.vad_merge_gap_var = tk.IntVar(value=200)
        self.overlap_var = tk.DoubleVar(value=0.30)
        self.min_segment_var = tk.DoubleVar(value=0.20)
        self.max_segment_var = tk.DoubleVar(value=4.0)
        self.hard_gap_var = tk.IntVar(value=60)
        self.timing_correction_var = tk.IntVar(value=0)
        self.batch_glob_var = tk.StringVar(value="**/*")
        self.batch_concurrency_var = tk.IntVar(value=1)
        self.log_level_var = tk.StringVar(value="INFO")

        self._build_ui()
        self.after(120, self._poll_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        basic = ttk.LabelFrame(root, text="Basic Mode", padding=10)
        basic.pack(fill="x")
        self._row(basic, 0, "Preset", ttk.Combobox(basic, textvariable=self.preset_var, values=list(PRESETS.keys()), state="readonly", width=18))
        self._row(basic, 1, "Model", ttk.Combobox(basic, textvariable=self.model_var, values=["small", "medium", "large-v2", "large-v3-turbo"], width=20))
        self._row(basic, 2, "GPU", ttk.Combobox(basic, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly", width=10))

        toggle = ttk.Frame(root)
        toggle.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(toggle, text="Show Advanced", variable=self.show_advanced_var, command=self._toggle_advanced).pack(side="left")

        mode = ttk.LabelFrame(root, text="Input", padding=10)
        mode.pack(fill="x", pady=(10, 0))
        ttk.Radiobutton(mode, text="Single File", variable=self.mode_var, value="single", command=self._switch_mode).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode, text="Batch Folder", variable=self.mode_var, value="batch", command=self._switch_mode).grid(row=0, column=1, sticky="w", padx=(20, 0))

        self.single_frame = ttk.Frame(mode)
        self.single_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self._build_single_frame(self.single_frame)

        self.batch_frame = ttk.Frame(mode)
        self.batch_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self._build_batch_frame(self.batch_frame)

        self.advanced = ttk.LabelFrame(root, text="Advanced Mode", padding=10)
        self._build_advanced(self.advanced)

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=(10, 0))
        self.run_button = ttk.Button(actions, text="Run", command=self._start)
        self.run_button.pack(side="left")
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=(12, 0))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(actions, textvariable=self.status_var).pack(side="left", padx=(12, 0))

        out = ttk.LabelFrame(root, text="Output", padding=10)
        out.pack(fill="both", expand=True, pady=(10, 0))
        self.output_text = tk.Text(out, height=20)
        self.output_text.pack(fill="both", expand=True)

        self.preset_var.trace_add("write", lambda *_: self._apply_preset_to_fields())
        self._apply_preset_to_fields()
        self._switch_mode()
        self._toggle_advanced()

    def _build_single_frame(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Input File").grid(row=0, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.single_input_var, width=90).grid(row=0, column=1, sticky="ew", padx=(10, 6))
        ttk.Button(parent, text="Browse", command=self._browse_single_input).grid(row=0, column=2, sticky="w")
        ttk.Label(parent, text="Output SRT").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=self.single_output_var, width=90).grid(row=1, column=1, sticky="ew", padx=(10, 6), pady=(6, 0))
        ttk.Button(parent, text="Browse", command=self._browse_single_output).grid(row=1, column=2, sticky="w", pady=(6, 0))

    def _build_batch_frame(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Input Folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.batch_input_var, width=90).grid(row=0, column=1, sticky="ew", padx=(10, 6))
        ttk.Button(parent, text="Browse", command=self._browse_batch_input).grid(row=0, column=2, sticky="w")
        ttk.Label(parent, text="Output Folder").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=self.batch_output_var, width=90).grid(row=1, column=1, sticky="ew", padx=(10, 6), pady=(6, 0))
        ttk.Button(parent, text="Browse", command=self._browse_batch_output).grid(row=1, column=2, sticky="w", pady=(6, 0))

    def _build_advanced(self, parent: ttk.LabelFrame) -> None:
        self._row(parent, 0, "Alignment", ttk.Combobox(parent, textvariable=self.alignment_var, values=["on", "off"], state="readonly", width=10))
        self._row(parent, 1, "Align Device", ttk.Combobox(parent, textvariable=self.align_device_var, values=["cuda", "cpu"], state="readonly", width=10))
        self._row(parent, 2, "Language", ttk.Entry(parent, textvariable=self.language_var, width=12))
        self._row(parent, 3, "Global Shift(ms)", ttk.Spinbox(parent, from_=-2000, to=2000, textvariable=self.shift_var, width=12))
        self._row(parent, 4, "VAD Threshold", ttk.Spinbox(parent, from_=0.1, to=0.9, increment=0.01, textvariable=self.vad_threshold_var, width=12))
        self._row(parent, 5, "Min Speech(ms)", ttk.Spinbox(parent, from_=100, to=2000, textvariable=self.vad_min_speech_var, width=12))
        self._row(parent, 6, "Min Silence(ms)", ttk.Spinbox(parent, from_=50, to=2000, textvariable=self.vad_min_silence_var, width=12))
        self._row(parent, 7, "Merge Gap(ms)", ttk.Spinbox(parent, from_=0, to=2000, textvariable=self.vad_merge_gap_var, width=12))
        self._row(parent, 8, "Min Segment(sec)", ttk.Spinbox(parent, from_=0.1, to=3.0, increment=0.05, textvariable=self.min_segment_var, width=12))
        self._row(parent, 9, "Max Segment(sec)", ttk.Spinbox(parent, from_=1.0, to=10.0, increment=0.1, textvariable=self.max_segment_var, width=12))
        self._row(parent, 10, "Overlap(sec)", ttk.Spinbox(parent, from_=0.0, to=1.0, increment=0.05, textvariable=self.overlap_var, width=12))
        self._row(parent, 11, "Timing Correction(ms)", ttk.Spinbox(parent, from_=0, to=300, textvariable=self.timing_correction_var, width=12))
        self._row(parent, 12, "Hard Gap(ms)", ttk.Spinbox(parent, from_=0, to=500, textvariable=self.hard_gap_var, width=12))
        self._row(parent, 13, "Batch Glob", ttk.Entry(parent, textvariable=self.batch_glob_var, width=30))
        self._row(parent, 14, "Batch Concurrency", ttk.Spinbox(parent, from_=1, to=8, textvariable=self.batch_concurrency_var, width=12))
        self._row(parent, 15, "Log Level", ttk.Combobox(parent, textvariable=self.log_level_var, values=["DEBUG", "INFO", "WARNING", "ERROR"], state="readonly", width=12))

    def _row(self, parent: ttk.Widget, row: int, label: str, widget: ttk.Widget) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        widget.grid(row=row, column=1, sticky="w", pady=2, padx=(10, 0))

    def _toggle_advanced(self) -> None:
        if self.show_advanced_var.get():
            self.advanced.pack(fill="x", pady=(10, 0))
        else:
            self.advanced.pack_forget()

    def _switch_mode(self) -> None:
        if self.mode_var.get() == "single":
            self.batch_frame.grid_remove()
            self.single_frame.grid()
        else:
            self.single_frame.grid_remove()
            self.batch_frame.grid()

    def _apply_preset_to_fields(self) -> None:
        values = PRESETS.get(self.preset_var.get(), {})
        if "model" in values:
            self.model_var.set(str(values["model"]))
        if "language" in values:
            self.language_var.set(str(values["language"]))
        if "alignment" in values:
            self.alignment_var.set(str(values["alignment"]))
        if "global_shift_ms" in values:
            self.shift_var.set(int(values["global_shift_ms"]))
        if "vad_threshold" in values:
            self.vad_threshold_var.set(float(values["vad_threshold"]))
        if "vad_min_speech_ms" in values:
            self.vad_min_speech_var.set(int(values["vad_min_speech_ms"]))
        if "vad_min_silence_ms" in values:
            self.vad_min_silence_var.set(int(values["vad_min_silence_ms"]))
        if "vad_pad_ms" in values:
            self.vad_pad_var.set(int(values["vad_pad_ms"]))
        if "vad_merge_gap_ms" in values:
            self.vad_merge_gap_var.set(int(values["vad_merge_gap_ms"]))
        if "beam_size" in values:
            pass
        if "overlap_sec" in values:
            self.overlap_var.set(float(values["overlap_sec"]))
        if "timing_correction_ms" in values:
            self.timing_correction_var.set(int(values["timing_correction_ms"]))

    def _browse_single_input(self) -> None:
        path = filedialog.askopenfilename(title="Select media file")
        if path:
            self.single_input_var.set(path)
            if not self.single_output_var.get():
                self.single_output_var.set(str(Path(path).with_suffix(".srt")))

    def _browse_single_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".srt", filetypes=[("SRT", "*.srt")])
        if path:
            self.single_output_var.set(path)

    def _browse_batch_input(self) -> None:
        path = filedialog.askdirectory(title="Select input folder")
        if path:
            self.batch_input_var.set(path)
            if not self.batch_output_var.get():
                self.batch_output_var.set(path)

    def _browse_batch_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.batch_output_var.set(path)

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            self._validate()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.output_text.delete("1.0", "end")
        self.run_button.configure(state="disabled")
        self.progress.start(10)
        self.status_var.set("Running...")
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def _validate(self) -> None:
        if self.mode_var.get() == "single":
            path = Path(self.single_input_var.get().strip())
            if not path.exists():
                raise ValueError("Single input file not found.")
        else:
            path = Path(self.batch_input_var.get().strip())
            if not path.exists() or not path.is_dir():
                raise ValueError("Batch input folder not found.")

    def _run_worker(self) -> None:
        try:
            if self.mode_var.get() == "single":
                report = run_single_with_runtime(self._build_single_config(), self._build_runtime(), keep_wav=False)
                self.queue.put(("log", self._format_report(report)))
            else:
                summary = run_batch_with_runtime(self._build_batch_configs(), self._build_runtime(), keep_wav=False)
                self.queue.put(("log", self._format_batch_summary(summary)))
            self.queue.put(("done", "Completed"))
        except SubgenError as exc:
            self.queue.put(("error", exc.user_message))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _build_runtime(self) -> RuntimeConfig:
        return RuntimeConfig(
            log_level=self.log_level_var.get(),
            batch_concurrency=max(1, int(self.batch_concurrency_var.get())),
        )

    def _common_config_kwargs(self) -> dict:
        return {
            "sample_rate": 16000,
            "global_shift_ms": int(self.shift_var.get()),
            "vad": VADConfig(
                threshold=float(self.vad_threshold_var.get()),
                min_speech_ms=int(self.vad_min_speech_var.get()),
                min_silence_ms=int(self.vad_min_silence_var.get()),
                pad_ms=int(self.vad_pad_var.get()),
                merge_gap_ms=int(self.vad_merge_gap_var.get()),
            ),
            "transcription": TranscriptionConfig(
                model_size=self.model_var.get().strip() or "large-v2",
                device=self.device_var.get(),
                compute_type="float16",
                beam_size=int(PRESETS.get(self.preset_var.get(), {}).get("beam_size", 8)),
                language=self.language_var.get().strip() or None,
                overlap_sec=float(self.overlap_var.get()),
            ),
            "alignment": AlignmentConfig(
                enabled=(self.alignment_var.get() == "on"),
                device=self.align_device_var.get(),
            ),
            "timing": TimingConfig(
                min_duration_sec=float(self.min_segment_var.get()),
                max_duration_sec=float(self.max_segment_var.get()),
                hard_gap_sec=float(self.hard_gap_var.get()) / 1000.0,
                onset_nudge_ms=int(self.timing_correction_var.get()),
            ),
        }

    def _build_single_config(self) -> PipelineConfig:
        input_path = Path(self.single_input_var.get().strip()).resolve()
        output_path = Path(self.single_output_var.get().strip()).resolve()
        temp_wav = Path(tempfile.gettempdir()) / f"{input_path.stem}.subgen.16k.wav"
        return PipelineConfig(input_path=input_path, output_path=output_path, temp_wav_path=temp_wav, **self._common_config_kwargs())

    def _build_batch_configs(self) -> list[PipelineConfig]:
        input_dir = Path(self.batch_input_var.get().strip()).resolve()
        output_dir = Path(self.batch_output_var.get().strip()).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            p for p in input_dir.glob(self.batch_glob_var.get().strip() or "**/*")
            if p.is_file() and p.suffix.lower() in MEDIA_EXTS
        )
        if not files:
            raise RuntimeError("No media files found in batch input folder.")

        configs: list[PipelineConfig] = []
        for src in files:
            rel = src.relative_to(input_dir)
            out_srt = (output_dir / rel).with_suffix(".srt")
            out_srt.parent.mkdir(parents=True, exist_ok=True)
            temp_wav = Path(tempfile.gettempdir()) / f"{src.stem}.subgen.16k.wav"
            configs.append(PipelineConfig(input_path=src, output_path=out_srt, temp_wav_path=temp_wav, **self._common_config_kwargs()))
        return configs

    def _format_report(self, report) -> str:
        lines = [
            str(report.output_path),
            f"  audio={report.audio_duration_sec:.1f}s speech_regions={report.speech_segment_count} avg_speech={report.avg_speech_segment_sec:.2f}s short_speech={report.short_speech_segment_count}",
            f"  rough_segments={report.rough_segment_count} final_segments={report.segment_count} avg_final={report.avg_final_segment_sec:.2f}s short_final={report.short_final_segment_ratio:.2%}",
            f"  alignment={'applied' if report.alignment_applied else 'skipped'} transcription_device={report.transcription_device} avg_shift={report.alignment_avg_abs_shift_ms:.1f}ms stddev={report.alignment_onset_stddev_ms:.1f}ms",
        ]
        if report.alignment_warning:
            lines.append(f"  alignment_warning={report.alignment_warning}")
        if report.transcription_warning:
            lines.append(f"  transcription_warning={report.transcription_warning}")
        if report.log_path:
            lines.append(f"  log={report.log_path}")
        return "\n".join(lines)

    def _format_batch_summary(self, summary) -> str:
        lines = [
            f"Batch Summary: total={len(summary.items)} ok={summary.succeeded} fail={summary.failed} requested_workers={summary.requested_concurrency} effective_workers={summary.effective_concurrency}"
        ]
        for item in summary.items:
            if item.success and item.report is not None:
                lines.append(
                    f"  OK {item.input_path.name} -> {item.output_path.name} "
                    f"segments={item.report.segment_count} avg_shift={item.report.alignment_avg_abs_shift_ms:.1f}ms"
                )
            else:
                lines.append(
                    f"  FAIL {item.input_path.name} category={item.error_category} message={item.user_message}"
                )
                if item.log_path is not None:
                    lines.append(f"    log={item.log_path}")
        return "\n".join(lines)

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, msg = self.queue.get_nowait()
                if kind == "log":
                    self.output_text.insert("end", msg + "\n")
                    self.output_text.see("end")
                elif kind == "done":
                    self.status_var.set(msg)
                    self.progress.stop()
                    self.run_button.configure(state="normal")
                elif kind == "error":
                    self.status_var.set("Failed")
                    self.progress.stop()
                    self.run_button.configure(state="normal")
                    messagebox.showerror("Error", msg)
        except Empty:
            pass
        self.after(120, self._poll_queue)


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
