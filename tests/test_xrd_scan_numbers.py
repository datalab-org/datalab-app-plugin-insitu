import zipfile
from pathlib import Path

import pytest

from datalab_app_plugin_insitu.apps.xrd.utils import extract_scan_numbers

EXAMPLE_ZIP = Path(__file__).parent.parent / "example_data" / "xrd" / "Example_in_situ_XRD_data.zip"


@pytest.mark.parametrize(
    "filenames, expected",
    [
        # Original 7-digit beamline format
        (["1058064-mythen_summed.dat", "1058063-mythen_summed.dat"], [1058064, 1058063]),
        # Constant trailing digit (mythen3) is ignored
        (["1355640_summed_mythen3.xye", "1355639_summed_mythen3.xye"], [1355640, 1355639]),
        # Short zero-padding
        (["0000-scan.dat", "0001-scan.dat", "0010-scan.dat"], [0, 1, 10]),
        # No padding at all
        (["scan_1.xy", "scan_2.xy", "scan_10.xy", "scan_100.xy"], [1, 2, 10, 100]),
        # Mixed padding
        (["scan_000001.xy", "scan_0000003.xy", "scan_0000112.xy"], [1, 3, 112]),
        # Constant sample number before the scan number
        (["sample2_scan_7.xy", "sample2_scan_8.xy"], [7, 8]),
        # Repeated (non-unique) date before the scan number
        (
            ["20250702_1.xy", "20250702_2.xy", "20250703_3.xy"],
            [1, 2, 3],
        ),
        # Digits in the extension are not considered
        (["scan_1.h5", "scan_2.h5"], [1, 2]),
        # Single file: longest run of digits is used
        (["1355639_summed_mythen3.xye"], [1355639]),
        ([], []),
    ],
)
def test_extract_scan_numbers_auto(filenames, expected):
    assert extract_scan_numbers(filenames) == expected


def test_extract_scan_numbers_inconsistent_runs():
    # Different numbers of digit runs per file: fall back to the longest run in each
    assert extract_scan_numbers(["scan_12.xy", "scan_13_retry2.xy"]) == [12, 13]


def test_extract_scan_numbers_pattern():
    filenames = ["run5_scan3.xy", "run5_scan4.xy", "run6_scan1.xy"]
    assert extract_scan_numbers(filenames, pattern=r"scan(\d+)") == [3, 4, 1]
    assert extract_scan_numbers(["a_5.xy", "a_6.xy"], pattern=r"\d+") == [5, 6]

    with pytest.raises(ValueError, match="Could not find a scan number matching"):
        extract_scan_numbers(["scan_1.xy", "other.xy"], pattern=r"scan_(\d+)")


def test_extract_scan_numbers_errors():
    with pytest.raises(ValueError, match="Could not find a scan number"):
        extract_scan_numbers(["scan_1.xy", "background.xy"])

    # Raw and summed files for the same scan without a glob to separate them
    with pytest.raises(ValueError, match="not unique"):
        extract_scan_numbers(
            ["1058063.dat", "1058063-mythen_1.dat", "1058064.dat", "1058064-mythen_1.dat"]
        )


@pytest.fixture
def unpadded_xrd_folder(tmp_path):
    """Rewrite the example temperature-mode data with unpadded scan numbers starting from 0."""
    offset = 1058063
    xrd_dir = tmp_path / "data" / "scans"
    log_dir = tmp_path / "data" / "log"
    xrd_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)

    with zipfile.ZipFile(EXAMPLE_ZIP) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX"):
                continue
            path = Path(name)
            if path.parent.name == "43_up" and path.name.endswith("-mythen_summed.dat"):
                scan = int(path.name.split("-")[0]) - offset
                (xrd_dir / f"scan_{scan}.dat").write_bytes(zf.read(name))
            elif path.parent.name == "log" and path.suffix == ".csv":
                lines = zf.read(name).decode().splitlines()
                rewritten = [lines[0]]
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    scan, temp = line.split(",", 1)
                    rewritten.append(f"{int(scan) - offset},{temp}")
                (log_dir / "log.csv").write_text("\n".join(rewritten) + "\n")

    return tmp_path / "data"


def test_unpadded_scan_numbers_temperature_mode(unpadded_xrd_folder):
    from datalab_app_plugin_insitu.apps.xrd.utils import process_local_xrd_data

    data = process_local_xrd_data(
        file_path=unpadded_xrd_folder,
        xrd_folder_name=Path("scans"),
        log_folder_name=Path("log"),
        time_series_source="log",
    )

    num_experiments = 52
    assert data["2D_data"].shape[0] == num_experiments
    assert list(data["2D_data"].index) == list(range(num_experiments))
    assert data["file_num_index"][0, 0] == 0
    assert data["file_num_index"][-1, 0] == num_experiments - 1
    assert data["index_df"]["file_num"].iloc[0] == 0
    assert data["index_df"]["file_num"].iloc[-1] == num_experiments - 1
    assert data["index_df"]["Temperature"].iloc[0] == pytest.approx(23.95)
    assert data["index_df"]["Temperature"].iloc[-1] == pytest.approx(593.35)


def test_scan_number_pattern_block_option(unpadded_xrd_folder):
    from datalab_app_plugin_insitu.apps.xrd.blocks import XRDInsituBlock

    block = XRDInsituBlock(item_id="test-xrd-insitu-pattern")
    block.data["time_series_source"] = "log"
    block.data["xrd_folder_name"] = "scans"
    block.data["time_series_folder_name"] = "log"
    block.data["scan_number_pattern"] = r"scan_(\d+)\.dat"
    data = block.process_and_store_data(unpadded_xrd_folder)

    assert list(data["2D_data"].index) == list(range(52))
