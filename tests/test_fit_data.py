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
        file1_path = "/Users/Ben/Desktop/datalab-app-plugin-nmr-insitu/example_data/Example-TEGDME/LiLiTEGDMEinsitu_02/LiLiTEGDMEinsitu_02.rds"
        file2_path = "/Users/Ben/Desktop/datalab-app-plugin-nmr-insitu/example_data/LiLiTEGDMEinsitu_02/LiLiTEGDMEinsitu_02_df_all.json"

        self.file_1_data = pyreadr.read_r(file1_path)
        self.file_1_df = self.file_1_data[None]
        self.file_2_df = pd.read_json(file2_path)

        self.file_1_total_df = self.file_1_df[self.file_1_df['peak']
                                              == 'Total intensity']
        self.file_1_peak1_df = self.file_1_df[self.file_1_df['peak'] == 'Peak 1']
        self.file_1_peak2_df = self.file_1_df[self.file_1_df['peak'] == 'Peak 2']

        self.file_1_reconstructed_df = {
            "data_df": self.file_1_total_df[['time', 'intensity', 'norm_intensity']],
            "df_peakfit1": self.file_1_peak1_df[['time', 'intensity', 'norm_intensity']],
            "df_peakfit2": self.file_1_peak2_df[['time', 'intensity', 'norm_intensity']]
        }

        for key, df in self.file_1_reconstructed_df.items():
            df['time'] = df['time'] / 3600

    def test_columns_exist(self):
        expected_arrays = ['data_df', 'df_peakfit1', 'df_peakfit2']
        expected_columns = ['time', 'intensity', 'norm_intensity']

        for array in expected_arrays:
            self.assertTrue(array in self.file_1_reconstructed_df,
                            f"{array} is missing in file_1_df")
            self.assertTrue(array in self.file_2_df,
                            f"{array} is missing in file_2_df")
            for column in expected_columns:
                self.assertTrue(column in self.file_1_reconstructed_df[array],
                                f"{column} is missing in {array} of file_1_df")
                self.assertTrue(column in self.file_2_df[array],
                                f"{column} is missing in {array} of file_2_df")

    def test_column_values(self):
        arrays = ['data_df', 'df_peakfit1', 'df_peakfit2']
        columns = ['time', 'intensity', 'norm_intensity']
        stats = ['min', 'max', 'mean']

        for array in arrays:
            for column in columns:
                for stat in stats:
                    with self.subTest(array=array, column=column, stat=stat):
                        file1_stat = getattr(
                            self.file_1_reconstructed_df[array][column], stat)()
                        file2_stat = getattr(
                            pd.Series(self.file_2_df[array][column]), stat)()

                        self.assertLessEqual(
                            percentage_difference(file1_stat, file2_stat),
                            TOLERANCE_PERCENT,
                            f"Difference greater than {TOLERANCE_PERCENT}% for {stat} of {column} (file1_stat: {file1_stat} & file2_stat: {file2_stat})"
                        )

        random_indices = random.sample(
            range(len(self.file_1_reconstructed_df)), 3)
        for index in random_indices:
            with self.subTest(index=index):
                for array in arrays:
                    for column in columns:
                        self.assertLessEqual(
                            percentage_difference(
                                self.file_1_reconstructed_df[array][column][index], self.file_2_df[array][column][index]),
                            TOLERANCE_PERCENT,
                            f"Difference greater than {TOLERANCE_PERCENT}% for index {index} in column {column}"
                        )


if __name__ == '__main__':
    unittest.main()
