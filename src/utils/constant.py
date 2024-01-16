from pathlib import Path

curr_path = Path(__file__).parents[1]
curr_path1 = Path(__file__).parents[2]
data_directory = curr_path / "data"

data_path_mall_customers = data_directory / 'mall_customers.csv'