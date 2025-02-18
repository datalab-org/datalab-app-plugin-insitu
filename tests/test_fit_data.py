import pandas as pd

#! Disable test_fit because we don't use fit_peaks for now.
#! So there's is no fit data inside result
#! To make test work
TOLERANCE_PERCENT = 50


# def test_columns_exist(get_tests_data):

#     result, _, fit_peaks = get_tests_data
#     df = pd.DataFrame(result['df'])

#     expected_arrays = ['data_df', 'df_peakfit1', 'df_peakfit2']
#     expected_columns = ['time', 'intensity', 'norm_intensity']

#     for array in expected_arrays:
#         assert array in df_fit, f"{array} is missing in df_fit"
#         assert array in processed_df_fit, f"{array} is missing in processed_df_fit"
#         for column in expected_columns:
#             assert column in df_fit[array], f"{column} is missing in df_fit"
#             assert column in processed_df_fit[array], f"{column} is missing in processed_df_fit"


# def test_column_values(get_tests_data, percentage_difference):

#     result, _, fit_peaks = get_tests_data
#     df = pd.DataFrame(result['df'])

#     arrays = ['data_df', 'df_peakfit1', 'df_peakfit2']
#     columns = ['time', 'intensity', 'norm_intensity']
#     stats = ['min', 'max', 'mean']

#     for array in arrays:
#         for column in columns:
#             for stat in stats:
#                 df_fit_stat = getattr(
#                     df_fit[array][column], stat)()
#                 processed_fit_stat = getattr(
#                     pd.Series(processed_df_fit[array][column]), stat)()

#                 assert percentage_difference(df_fit_stat, processed_fit_stat) <= TOLERANCE_PERCENT, \
#                     f"Difference greater than {TOLERANCE_PERCENT}% for {stat} of {column} (df_fit: {df_fit_stat} & processed_df_fit: {processed_fit_stat})"
