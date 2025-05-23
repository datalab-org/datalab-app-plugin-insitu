import pandas as pd

TOLERANCE_PERCENT = 0.01

#! These test doesn't make sense anymore since we don't use or display these data in the insitu NMR datablock


def test_columns_exist(get_tests_data):
    result = get_tests_data
    df = pd.DataFrame(result["integrated_data"])

    expected_columns = ["time", "intensity", "norm_intensity"]

    for column in expected_columns:
        assert column in df.columns, f"{column} is missing in df"
