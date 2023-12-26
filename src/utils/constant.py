from pathlib import Path


curr_path = Path(__file__).parents[1]
data_directory = curr_path / "data"
house_dataset = data_directory/'housing_price_dataset.csv'
titanic_train_x = data_directory/'train_X.csv'
titanic_train_y = data_directory/'train_Y.csv'
titanic_test_x = data_directory/'test_X.csv'
titanic_test_y = data_directory/'test_Y.csv'
