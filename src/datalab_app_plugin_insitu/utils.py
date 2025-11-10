import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd


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


def flexible_data_reader(
    file_path: Union[str, Path],
    separators: Optional[list[str]] = None,
    required_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Flexible data reader that can handle different file formats and delimiters.

    Tries to intelligently read CSV, TXT, and Excel files by:
    1. Detecting Excel formats and using pd.read_excel()
    2. Trying pd.read_csv() with sep=None (auto-detection) using python engine
    3. Falling back to trying specific separators if provided

    Args:
        file_path: Path to the data file (csv, txt, xlsx, etc.)
        separators: Optional list of separators to try (e.g., [",", "\t", r"\\s+"])
                   If None, uses pandas' auto-detection first
        required_columns: Optional list of column names that must be present

    Returns:
        pd.DataFrame: The loaded dataframe

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file cannot be parsed or required columns are missing
    """
    EXCEL_LIKE_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"}

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    # Handle Excel-like formats
    if file_path.suffix.lower() in EXCEL_LIKE_EXTENSIONS:
        try:
            df_dict = pd.read_excel(file_path, sheet_name=None)
            df = next(iter(df_dict.values()))

            if len(df_dict) > 1:
                import warnings

                warnings.warn(
                    f"Found {len(df_dict)} sheets in {file_path.name}, using only the first one."
                )
        except Exception as e:
            raise ValueError(f"Failed to read Excel file {file_path}. Error: {e}") from e
    else:
        # Try pandas auto-detection first (works for most well-formed CSV/TSV files)
        try:
            df = pd.read_csv(
                file_path,
                sep=None,  # Auto-detect separator
                encoding_errors="backslashreplace",
                engine="python",
            )

            # Check if we got a reasonable result
            if len(df.columns) > 1 and not df.empty:
                # Successfully read with auto-detection
                pass
            else:
                raise ValueError("Auto-detection resulted in single column or empty dataframe")

        except Exception as auto_detect_error:
            # If auto-detection fails, try specific separators
            if separators is None:
                separators = [",", "\t", r"\s+", ";"]

            df = None
            errors = []

            for sep in separators:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding_errors="backslashreplace")

                    # Validate we got reasonable data
                    if len(df.columns) > 1 and not df.empty:
                        break
                    else:
                        errors.append(f"sep='{sep}': Single column or empty result")
                        df = None

                except Exception as e:
                    errors.append(f"sep='{sep}': {type(e).__name__}")
                    continue

            if df is None:
                error_summary = "; ".join(errors)
                raise ValueError(
                    f"Failed to parse file {file_path}. "
                    f"Tried separators: {separators}. "
                    f"Errors: {error_summary}. "
                    f"Original auto-detect error: {auto_detect_error}"
                ) from auto_detect_error

    # Validate required columns if specified
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"File {file_path} is missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

    return df
