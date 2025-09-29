#!/usr/bin/env python3
"""Create a self-extracting archive of the repository."""

from __future__ import annotations

import argparse
import base64
import io
import os
import stat
import zipfile
from pathlib import Path
from typing import Iterable

EXCLUDE_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    "dist",
    "node_modules",
    ".venv",
}

EXCLUDE_FILES = {
    ".DS_Store",
}

STUB_TEMPLATE = """#!/usr/bin/env python3
\"\"\"Deep-Live-Cam self-extracting archive.\"\"\"

import base64
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

ENCODED_ARCHIVE = \"\"\"{encoded_data}\"\"\"


def main() -> None:
    archive_bytes = base64.b64decode("".join(ENCODED_ARCHIVE.split()))
    target_dir = tempfile.mkdtemp(prefix="deeplivecam-")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        zf.extractall(target_dir)

    repo_root = Path(target_dir)
    message = (
        "Deep-Live-Cam repository extracted to: {{path}}\n\n"
        "You can now run the application with:\n"
        "    python run.py\n"
        "(Executed from the extracted repository directory.)"
    ).format(path=repo_root)
    print(message)

    run_py = repo_root / "run.py"
    if run_py.exists():
        print("Launching run.py ...")
        os.execv(sys.executable, [sys.executable, str(run_py)])
    else:
        print("run.py not found in the extracted repository. Exiting without execution.")


if __name__ == "__main__":
    main()
"""


def iter_files(base_path: Path) -> Iterable[Path]:
    for path in base_path.rglob("*"):
        if path.is_dir():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.name in EXCLUDE_FILES:
            continue
        yield path


def make_self_extracting_archive(output: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in iter_files(repo_root):
            archive.write(file_path, file_path.relative_to(repo_root))

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    formatted = "\n".join(
        encoded[i : i + 76] for i in range(0, len(encoded), 76)
    )
    stub = STUB_TEMPLATE.format(encoded_data=formatted)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stub)
    current_mode = output.stat().st_mode
    output.chmod(current_mode | stat.S_IEXEC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a self-extracting Deep-Live-Cam archive."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/Deep-Live-Cam.sfx.py"),
        help="Where to write the self-extracting archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_self_extracting_archive(args.output)
    print(f"Self-extracting archive written to {args.output}")


if __name__ == "__main__":
    main()
