from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Dict


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_app_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[1]


def get_exe_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return get_app_root()


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".ararat_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def get_writable_root() -> Path:
    preferred = get_exe_dir()
    if _is_writable_directory(preferred):
        return preferred
    appdata = os.environ.get("APPDATA")
    if appdata:
        fallback = Path(appdata) / "ARARAT"
    else:
        fallback = Path.home() / "AppData" / "Roaming" / "ARARAT"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def resolve_path(*parts: str) -> Path:
    return get_app_root().joinpath(*parts)


def resolve_writable_path(*parts: str) -> Path:
    return get_writable_root().joinpath(*parts)


def get_config_path() -> Path:
    return resolve_writable_path("config_local.json")


def ensure_dirs() -> Dict[str, Path]:
    root = get_writable_root()
    exports = root / "exports"
    logs = root / "logs"
    data = root / "data"
    exports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return {
        "app_root": get_app_root(),
        "writable_root": root,
        "exports": exports,
        "logs": logs,
        "data": data,
    }


def init_logging(log_file_name: str = "ararat_viewer.log") -> Path:
    dirs = ensure_dirs()
    log_path = dirs["logs"] / log_file_name
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
    logger = logging.getLogger("ararat")
    logger.info("startup app_root=%s", dirs["app_root"])
    logger.info("startup writable_root=%s", dirs["writable_root"])
    logger.info("startup exports=%s", dirs["exports"])
    logger.info("startup logs=%s", dirs["logs"])
    logger.info("startup data=%s", dirs["data"])
    logger.info("startup python=%s", sys.version.replace("\n", " "))
    logger.info("startup platform=%s", platform.platform())
    return log_path


def show_error_popup(message: str, title: str = "ARARAT Viewer") -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        pass
