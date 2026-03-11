from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

from huggingface_hub import HfApi

T = TypeVar("T")


def retry(
    fn: Callable[[], T],
    attempts: int = 3,
    delay_seconds: float = 2.0,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_exceptions as error:
            last_error = error
            if attempt == attempts:
                raise
            time.sleep(delay_seconds * attempt)
    assert last_error is not None
    raise last_error


def resolve_model_revision(model_id: str, revision: str | None) -> str | None:
    try:
        model_info = retry(lambda: HfApi().model_info(model_id, revision=revision))
    except Exception:
        return revision
    return model_info.sha or revision


def resolved_revision_record(model_name: str, model_id: str, revision: str | None) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_id": model_id,
        "requested_revision": revision or "main",
        "resolved_revision": resolve_model_revision(model_id, revision),
    }

