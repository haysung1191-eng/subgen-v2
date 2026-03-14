from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_LOG_DIR_NAME = "logs"


@dataclass(slots=True)
class LogSession:
    logger: logging.Logger
    log_path: Path
    handler: logging.Handler

    def close(self) -> None:
        self.logger.removeHandler(self.handler)
        self.handler.close()


def create_log_session(log_dir: Path, stem: str, level: str = "INFO") -> LogSession:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem).strip("_") or "run"
    log_path = log_dir / f"{timestamp}-{safe_stem}.log"

    logger = logging.getLogger(f"subgen.run.{timestamp}.{safe_stem}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return LogSession(logger=logger, log_path=log_path, handler=handler)
