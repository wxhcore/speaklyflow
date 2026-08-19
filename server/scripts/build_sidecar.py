"""Build the SpeaklyFlow server as a PyInstaller onedir Tauri resource."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = PROJECT_ROOT / "server"
TAURI_ROOT = PROJECT_ROOT / "desktop" / "src-tauri"


def _copy_metadata_args(*distributions: str) -> list[str]:
    arguments: list[str] = []
    for distribution in distributions:
        try:
            importlib.metadata.distribution(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        arguments.extend(("--copy-metadata", distribution))
    return arguments


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("Desktop packaging currently supports macOS only")
    if importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit(
            "PyInstaller is missing; install the project build dependencies "
            "in the active Python environment"
        )

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/build_macos_voice_processing.py")],
        check=True,
    )

    machine = platform.machine().lower()
    target = "aarch64-apple-darwin" if machine == "arm64" else "x86_64-apple-darwin"
    binary_name = "speaklyflow-server"
    output_dir = TAURI_ROOT / "sidecar"
    build_dir = PROJECT_ROOT / "build" / "sidecar" / target
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["PYINSTALLER_CONFIG_DIR"] = str(build_dir / "cache")
    data_separator = os.pathsep
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        binary_name,
        "--distpath",
        str(output_dir),
        "--workpath",
        str(build_dir / "work"),
        "--specpath",
        str(build_dir),
        "--paths",
        str(PROJECT_ROOT / "src"),
        "--paths",
        str(SERVER_ROOT / "src"),
        "--collect-all",
        "sherpa_onnx",
        "--collect-binaries",
        "numpy",
        "--collect-submodules",
        "bumblehive",
        "--collect-data",
        "bumblehive",
        "--collect-submodules",
        "uvicorn",
        "--collect-submodules",
        "websockets",
        "--add-data",
        f"{PROJECT_ROOT / 'models'}{data_separator}models",
        "--add-data",
        (
            f"{PROJECT_ROOT / 'src/speaklyflow/vad/data'}"
            f"{data_separator}speaklyflow/vad/data"
        ),
        "--add-data",
        (
            f"{PROJECT_ROOT / 'src/speaklyflow/audio/_native/libSpeaklyFlowVoiceIO.dylib'}"
            f"{data_separator}speaklyflow/audio/_native"
        ),
        "--hidden-import",
        "bumblehive.agent.context.prompts",
        *_copy_metadata_args("bumblehive", "fastmcp", "fastmcp-slim"),
        str(SERVER_ROOT / "sidecar.py"),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)

    executable = output_dir / binary_name / binary_name
    if not executable.is_file():
        raise RuntimeError(f"Sidecar output missing: {executable}")
    print(f"Sidecar ready: {executable}")


if __name__ == "__main__":
    main()
