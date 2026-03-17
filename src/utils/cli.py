from __future__ import annotations

import sys
import traceback
from pathlib import Path


def format_compact_exception(
    error: BaseException,
    repo_root: str | Path,
    *,
    max_frames: int = 6,
) -> str:
    root = Path(repo_root).resolve()
    extracted = traceback.extract_tb(error.__traceback__)
    relevant = [
        frame
        for frame in extracted
        if "site-packages" not in frame.filename and "/lib/python" not in frame.filename
    ]
    frames = (relevant or extracted)[-max_frames:]

    lines = [f"{type(error).__name__}: {error}"]
    for frame in frames:
        frame_path = Path(frame.filename)
        try:
            location = str(frame_path.resolve().relative_to(root))
        except ValueError:
            location = frame.filename
        lines.append(f"  at {location}:{frame.lineno} in {frame.name}")
        if frame.line:
            lines.append(f"    {frame.line.strip()}")
    lines.append("  rerun with --full-traceback to show the complete stack")
    return "\n".join(lines)


def run_with_compact_errors(main_fn, repo_root: str | Path, *, full_traceback: bool = False) -> int:
    try:
        main_fn()
        return 0
    except KeyboardInterrupt:
        raise
    except Exception as error:
        if full_traceback:
            raise
        print(format_compact_exception(error, repo_root), file=sys.stderr)
        return 1
