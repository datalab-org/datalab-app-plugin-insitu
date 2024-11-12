import unittest
import pyreadr
import pandas as pd
import random

TOLERANCE_PERCENT = 2


def percentage_difference(val1, val2):
    if val1 == 0 or val2 == 0:
        return 0
    return abs(val1 - val2) / abs(val2) * 100


class TestDfDataComparison(unittest.TestCase):

    def setUp(self):
        file1_path = "/Users/Ben/Desktop/datalab-app-plugin-nmr-insitu/example_data/Example-TEGDME/LiLiTEGDMEinsitu_02/dfenv_LiLiTEGDMEinsitu_02.rds"
        file2_path = "/Users/Ben/Desktop/datalab-app-plugin-nmr-insitu/example_data/LiLiTEGDMEinsitu_02/dfenv_LiLiTEGDMEinsitu_02.json"

        self.file_1_data = pyreadr.read_r(file1_path)
        self.file_1_df = self.file_1_data[None]
        self.file_1_df['time'] = self.file_1_df['time'] / 3600
        self.file_2_df = pd.read_json(file2_path)

    def test_columns_exist(self):
        expected_columns = ['time', 'intensity', 'norm_intensity']

        for column in expected_columns:
            self.assertIn(column, self.file_1_df.columns,
                          f"{column} is missing in file_1_df")
            self.assertIn(column, self.file_2_df.columns,
                          f"{column} is missing in file_2_df")

    def test_column_values(self):
        columns = ['time', 'intensity', 'norm_intensity']
        stats = ['min', 'max', 'mean']

        for column in columns:
            for stat in stats:
                with self.subTest(column=column, stat=stat):
                    file1_stat = getattr(self.file_1_df[column], stat)()
                    file2_stat = getattr(self.file_2_df[column], stat)()

                    self.assertLessEqual(
                        percentage_difference(file1_stat, file2_stat),
                        TOLERANCE_PERCENT,
                        f"Difference greater than {TOLERANCE_PERCENT}% for {stat} of {column}"
                    )

        random_indices = random.sample(range(len(self.file_1_df)), 3)
        for index in random_indices:
            with self.subTest(index=index):
                for column in columns:
                    self.assertLessEqual(
                        percentage_difference(
                            self.file_1_df[column][index], self.file_2_df[column][index]),
                        TOLERANCE_PERCENT,
                        f"Difference greater than {TOLERANCE_PERCENT}% for index {index} in column {column}"
                    )


if __name__ == '__main__':
    unittest.main()
