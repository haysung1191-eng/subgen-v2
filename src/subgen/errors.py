from __future__ import annotations


class SubgenError(RuntimeError):
    def __init__(self, user_message: str, *, category: str, detail: str | None = None) -> None:
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.category = category
        self.detail = detail or user_message


class InputMediaError(SubgenError):
    def __init__(self, user_message: str, detail: str | None = None) -> None:
        super().__init__(user_message, category="input", detail=detail)


class FFmpegError(SubgenError):
    def __init__(self, detail: str) -> None:
        super().__init__("ffmpeg 오디오 추출에 실패했습니다.", category="ffmpeg", detail=detail)


class CUDAInitError(SubgenError):
    def __init__(self, detail: str) -> None:
        super().__init__("CUDA 초기화에 실패했습니다.", category="cuda", detail=detail)


class ModelLoadError(SubgenError):
    def __init__(self, detail: str) -> None:
        super().__init__("음성 인식 모델을 불러오지 못했습니다.", category="model", detail=detail)


class TranscriptionError(SubgenError):
    def __init__(self, detail: str) -> None:
        super().__init__("음성 인식 단계에서 실패했습니다.", category="transcription", detail=detail)


class OutputWriteError(SubgenError):
    def __init__(self, detail: str) -> None:
        super().__init__("자막 파일 저장에 실패했습니다.", category="output", detail=detail)


def classify_unexpected_error(exc: Exception) -> SubgenError:
    message = str(exc)
    lowered = message.lower()
    if "out of memory" in lowered or "cuda out of memory" in lowered:
        return SubgenError("메모리가 부족해 작업을 완료하지 못했습니다.", category="memory", detail=message)
    return SubgenError("처리 중 알 수 없는 오류가 발생했습니다.", category="runtime", detail=message)
