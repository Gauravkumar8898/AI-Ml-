from pathlib import Path


curr_path = Path(__file__).parents[1]
data_directory = curr_path / "data"
lung_cancer_dataset = data_directory/'cancer patient data sets.csv'