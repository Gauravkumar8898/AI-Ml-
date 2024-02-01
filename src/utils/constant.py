from pathlib import Path

curr_path = Path(__file__).parents[1]
data_directory = curr_path / "data"
x_train_data_path = data_directory / 'x_train.csv'
y_train_data_path = data_directory / 'y_train.csv'
x_test_data_path = data_directory / 'x_test.csv'
y_test_data_path = data_directory / 'y_test.csv'
