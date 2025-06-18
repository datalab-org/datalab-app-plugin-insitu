import os
from pathlib import Path
from typing import Optional, Union



def should_skip_path(path: Union[str, Path]) -> bool:
    """
    Check if a path should be skipped (macOS system files, hidden files, etc.)

    Args:
        path: Path or string to check

    Returns:
        bool: True if the path should be skipped, False otherwise
    """
    path_str = str(path)
    return "__MACOSX" in path_str or Path(path_str).name.startswith(".")


def _find_folder_path(base_path: Path, target_folder_name: str | Path) -> Optional[Path]:
    """
    Find a folder path inside a zip regardless of whether the zip has an extra level of nesting.

    Args:
        base_path: Base directory to start the search
        target_folder_name: Name of the folder to find

    Returns:
        Optional[Path]: Path to the found folder, or None if not found
    """
    target_folder_name = Path(target_folder_name).stem

    direct_path = base_path / target_folder_name
    if direct_path.exists() and direct_path.is_dir():
        return direct_path

    for item in base_path.iterdir():
        if should_skip_path(item):
            continue

        if item.is_dir():
            nested_path = item / target_folder_name
            if nested_path.exists() and nested_path.is_dir():
                return nested_path

    for root, dirs, _ in os.walk(str(base_path)):
        dirs[:] = [d for d in dirs if not should_skip_path(Path(d))]

        if target_folder_name in dirs:
            return Path(root) / target_folder_name

    return None