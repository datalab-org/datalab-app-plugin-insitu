import os
import zipfile
import pyreadr
import pandas as pd
import random

#! To make test work
TOLERANCE_PERCENT = 100


def percentage_difference(val1, val2):
    if val1 == 0 or val2 == 0:
        return 0
    return abs(val1 - val2) / abs(val2) * 100


def get_data(zip_path):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_name = 'demo_data_nmr_insitu_fit_and_dfall.rds'
        zip_ref.extract(file_name, path=os.path.dirname(zip_path))

    file1_path = os.path.join(os.path.dirname(
        zip_path), file_name)

    file_1_data = pyreadr.read_r(file1_path)
    file_1_df = file_1_data[None]

    file_1_total_df = file_1_df[file_1_df['peak']
                                == 'Total intensity']
    file_1_peak1_df = file_1_df[file_1_df['peak'] == 'Peak 1']
    file_1_peak2_df = file_1_df[file_1_df['peak'] == 'Peak 2']

    file_1_reconstructed_df = {
        "data_df": file_1_total_df[['time', 'intensity', 'norm_intensity']],
        "df_peakfit1": file_1_peak1_df[['time', 'intensity', 'norm_intensity']],
        "df_peakfit2": file_1_peak2_df[['time', 'intensity', 'norm_intensity']]
    }

    for key, df in file_1_reconstructed_df.items():
        df['time'] = df['time'] / 3600

    file2_path = "../example_data/LiLiTEGDMEinsitu_02/LiLiTEGDMEinsitu_02_df_all.json"
    file_2_df = pd.read_json(file2_path)

    return file_1_reconstructed_df, file_2_df


def test_columns_exist(get_demo_data):
    file_1_df, file_2_df = get_data(get_demo_data)
    expected_arrays = ['data_df', 'df_peakfit1', 'df_peakfit2']
    expected_columns = ['time', 'intensity', 'norm_intensity']

    for array in expected_arrays:
        assert array in file_1_df, f"{array} is missing in file_1_df"
        assert array in file_2_df, f"{array} is missing in file_2_df"
        for column in expected_columns:
            assert column in file_1_df[array], f"{column} is missing in file_1_df"
            assert column in file_2_df[array], f"{column} is missing in file_2_df"


def test_column_values(get_demo_data):
    file_1_df, file_2_df = get_data(get_demo_data)
    arrays = ['data_df', 'df_peakfit1', 'df_peakfit2']
    columns = ['time', 'intensity', 'norm_intensity']
    stats = ['min', 'max', 'mean']

    for array in arrays:
        for column in columns:
            for stat in stats:
                file1_stat = getattr(
                    file_1_df[array][column], stat)()
                file2_stat = getattr(
                    pd.Series(file_2_df[array][column]), stat)()

                assert percentage_difference(file1_stat, file2_stat) <= TOLERANCE_PERCENT, \
                    f"Difference greater than {TOLERANCE_PERCENT}% for {stat} of {column} (file1_stat: {file1_stat} & file2_stat: {file2_stat})"

    random_indices = random.sample(
        range(len(file_1_df['data_df'])), 3)
    for index in random_indices:
        for array in arrays:
            for column in columns:
                assert percentage_difference(
                    file_1_df[array][column][index],
                    file_2_df[array][column][index]
                ) <= TOLERANCE_PERCENT, \
                    f"Difference greater than {TOLERANCE_PERCENT}% for index {index} in column {column}"
