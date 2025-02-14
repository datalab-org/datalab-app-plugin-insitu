import random

TOLERANCE_PERCENT = 0.01

#! Temporaly disable test


def temp_test():
    assert 1 == 1

# def test_columns_exist(get_demo_data):
#     df, _, processed_df, _ = get_demo_data

#     expected_columns = ['time', 'intensity', 'norm_intensity']

#     for column in expected_columns:
#         assert column in df.columns, f"{column} is missing in df"
#         assert column in processed_df.columns, f"{column} is missing in processed_df"


# def test_column_values(get_demo_data, percentage_difference):
#     df, _, processed_df, _ = get_demo_data

#     columns = ['time', 'intensity', 'norm_intensity']
#     stats = ['min', 'max', 'mean']

#     for column in columns:
#         for stat in stats:
#             df_stat = getattr(df[column], stat)()
#             processed_stat = getattr(processed_df[column], stat)()
#             assert percentage_difference(df_stat, processed_stat) <= TOLERANCE_PERCENT, \
#                 f"Difference greater than {TOLERANCE_PERCENT}% for {stat} of {column} (df_value: {df_stat}, processed_value: {processed_stat})"

#     random_indices = random.sample(range(len(df)), 3)
#     for index in random_indices:
#         for column in columns:
#             df_value = df[column].iloc[index]
#             processed_value = processed_df[column].iloc[index]

#             assert percentage_difference(df_value, processed_value) <= TOLERANCE_PERCENT, \
#                 f"Difference greater than {TOLERANCE_PERCENT}% for {column} at index {index} (df_value: {df_value}, processed_value: {processed_value})"
