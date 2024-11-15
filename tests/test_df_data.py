import os
import zipfile
import pyreadr
import pandas as pd
import random

TOLERANCE_PERCENT = 0.01


def percentage_difference(val1, val2):
    if val1 == 0 or val2 == 0:
        return 0
    return abs(val1 - val2) / abs(val2) * 100


def get_data(zip_path):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_name = 'demo_data_nmr_insitu_df.rds'
        zip_ref.extract(file_name, path=os.path.dirname(zip_path))

    file1_path = os.path.join(os.path.dirname(
        zip_path), file_name)

    file_1_data = pyreadr.read_r(file1_path)
    file_1_df = file_1_data[None]
    file_1_df['time'] = file_1_df['time'] / 3600

    file2_path = "../example_data/LiLiTEGDMEinsitu_02/dfenv_LiLiTEGDMEinsitu_02.json"
    file_2_df = pd.read_json(file2_path)

    return file_1_df, file_2_df


def test_columns_exist(get_demo_data):
    file_1_df, file_2_df = get_data(get_demo_data)
    expected_columns = ['time', 'intensity', 'norm_intensity']

    for column in expected_columns:
        assert column in file_1_df.columns, f"{column} is missing in file_1_df"
        assert column in file_2_df.columns, f"{column} is missing in file_2_df"


def test_column_values(get_demo_data):
    file_1_df, file_2_df = get_data(get_demo_data)
    columns = ['time', 'intensity', 'norm_intensity']
    stats = ['min', 'max', 'mean']

    for column in columns:
        for stat in stats:
            file1_stat = getattr(file_1_df[column], stat)()
            file2_stat = getattr(file_2_df[column], stat)()

            assert percentage_difference(file1_stat, file2_stat) <= TOLERANCE_PERCENT, \
                f"Difference greater than {TOLERANCE_PERCENT}% for {stat} of {column}"

    random_indices = random.sample(range(len(file_1_df)), 3)
    for index in random_indices:
        for column in columns:
            assert percentage_difference(file_1_df[column][index], file_2_df[column][index]) <= TOLERANCE_PERCENT, \
                f"Difference greater than {TOLERANCE_PERCENT}% for index {index} in column {column}"
