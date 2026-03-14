from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig, RuntimeConfig
from .errors import SubgenError, classify_unexpected_error
from .logging_utils import DEFAULT_LOG_DIR_NAME, create_log_session
from .pipeline import PipelineReport, run_pipeline_with_report


@dataclass(slots=True)
class BatchItemResult:
    input_path: Path
    output_path: Path
    success: bool
    report: PipelineReport | None
    user_message: str | None = None
    debug_message: str | None = None
    error_category: str | None = None
    log_path: Path | None = None


@dataclass(slots=True)
class BatchRunSummary:
    items: list[BatchItemResult]
    requested_concurrency: int
    effective_concurrency: int

    @property
    def succeeded(self) -> int:
        return sum(1 for item in self.items if item.success)

    @property
    def failed(self) -> int:
        return sum(1 for item in self.items if not item.success)


def _default_log_dir(config: PipelineConfig, runtime: RuntimeConfig) -> Path:
    if runtime.log_dir is not None:
        return runtime.log_dir
    return config.output_path.parent / DEFAULT_LOG_DIR_NAME


def _effective_batch_concurrency(configs: list[PipelineConfig], requested: int) -> int:
    if requested <= 1:
        return 1
    for config in configs:
        if config.transcription.device == "cuda" or (config.alignment.enabled and config.alignment.device == "cuda"):
            return 1
    return requested


def _ensure_unique_output_path(target: Path) -> Path:
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    index = 1
    while True:
        candidate = parent / f"{stem}.{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def run_single_with_runtime(config: PipelineConfig, runtime: RuntimeConfig, keep_wav: bool = False) -> PipelineReport:
    log_session = create_log_session(_default_log_dir(config, runtime), config.input_path.stem, runtime.log_level)
    try:
        debug_export_path = None
        if runtime.debug_export_dir is not None:
            runtime.debug_export_dir.mkdir(parents=True, exist_ok=True)
            debug_export_path = runtime.debug_export_dir / f"{config.input_path.stem}.debug.json"
        report = run_pipeline_with_report(config, keep_wav=keep_wav, logger=log_session.logger, debug_export_path=debug_export_path)
        report.log_path = log_session.log_path
        return report
    finally:
        log_session.close()


def _run_batch_item(config: PipelineConfig, runtime: RuntimeConfig, keep_wav: bool) -> BatchItemResult:
    safe_output = _ensure_unique_output_path(config.output_path)
    config.output_path = safe_output
    log_session = create_log_session(_default_log_dir(config, runtime), config.input_path.stem, runtime.log_level)
    try:
        debug_export_path = None
        if runtime.debug_export_dir is not None:
            runtime.debug_export_dir.mkdir(parents=True, exist_ok=True)
            debug_export_path = runtime.debug_export_dir / f"{config.input_path.stem}.debug.json"
        report = run_pipeline_with_report(config, keep_wav=keep_wav, logger=log_session.logger, debug_export_path=debug_export_path)
        report.log_path = log_session.log_path
        return BatchItemResult(
            input_path=config.input_path,
            output_path=config.output_path,
            success=True,
            report=report,
            log_path=log_session.log_path,
        )
    except SubgenError as exc:
        log_session.logger.exception(exc.detail)
        return BatchItemResult(
            input_path=config.input_path,
            output_path=config.output_path,
            success=False,
            report=None,
            user_message=exc.user_message,
            debug_message=exc.detail,
            error_category=exc.category,
            log_path=log_session.log_path,
        )
    except Exception as exc:
        wrapped = classify_unexpected_error(exc)
        log_session.logger.exception(wrapped.detail)
        return BatchItemResult(
            input_path=config.input_path,
            output_path=config.output_path,
            success=False,
            report=None,
            user_message=wrapped.user_message,
            debug_message=wrapped.detail,
            error_category=wrapped.category,
            log_path=log_session.log_path,
        )
    finally:
        log_session.close()


def run_batch_with_runtime(
    configs: list[PipelineConfig],
    runtime: RuntimeConfig,
    keep_wav: bool = False,
) -> BatchRunSummary:
    effective = _effective_batch_concurrency(configs, runtime.batch_concurrency)
    items: list[BatchItemResult] = []
    if effective == 1:
        for config in configs:
            items.append(_run_batch_item(config, runtime, keep_wav))
        return BatchRunSummary(items=items, requested_concurrency=runtime.batch_concurrency, effective_concurrency=effective)

    with ThreadPoolExecutor(max_workers=effective) as executor:
        futures = [executor.submit(_run_batch_item, config, runtime, keep_wav) for config in configs]
        for future in as_completed(futures):
            items.append(future.result())

    items.sort(key=lambda item: str(item.input_path))
    return BatchRunSummary(items=items, requested_concurrency=runtime.batch_concurrency, effective_concurrency=effective)
