from __future__ import annotations

from datetime import datetime
from pathlib import Path

from openpyxl.workbook.workbook import Workbook

from core.paths import data_dir


OUTPUTS_DIR = data_dir("outputs")
BACKUPS_DIR = data_dir("backups")


def export_edited_workbook(workbook: Workbook, original_file_name: str) -> Path:
    """Save an edited workbook copy into outputs/ without overwriting the original."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    create_workbook_backup(workbook, original_file_name)

    original_path = Path(original_file_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    export_path = _unique_output_path(f"{original_path.stem}_edited_{timestamp}.xlsx")

    workbook.save(export_path)
    return export_path


def create_workbook_backup(workbook: Workbook, original_file_name: str) -> Path:
    """Save a local timestamped backup before edited export."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    original_path = Path(original_file_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = _unique_backup_path(f"{original_path.stem}_backup_{timestamp}.xlsx")
    workbook.save(backup_path)
    return backup_path


def _unique_output_path(file_name: str) -> Path:
    export_path = OUTPUTS_DIR / file_name
    if not export_path.exists():
        return export_path

    stem = export_path.stem
    suffix = export_path.suffix
    counter = 1
    while True:
        candidate = OUTPUTS_DIR / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _unique_backup_path(file_name: str) -> Path:
    backup_path = BACKUPS_DIR / file_name
    if not backup_path.exists():
        return backup_path

    stem = backup_path.stem
    suffix = backup_path.suffix
    counter = 1
    while True:
        candidate = BACKUPS_DIR / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
