from src.pipeline.titanic_pipeline import pipeline
from src.pipeline.mnist_pipeline import mnist_pipeline_
from src.pipeline.cats_dogs_pipeline import cats_dogs_pipeline
from src.pipeline.dogs_cats_transfer_learning import cats_dogs_tf_pipeline


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    inputs = abs(int(input("Enter Choice for mnist = 1 or titanic = 2 or cat-&-dog_CNN = 3 "
                           "or cat-&-dog_transfer = 4:")))
    if inputs == 2:
        pipeline()
    elif inputs == 1:
        mnist_pipeline_()
    elif inputs == 3:
        cats_dogs_pipeline()
    elif inputs == 4:
        cats_dogs_tf_pipeline()
    else:
        print("Enter  1 or 2 or 3 or 4 only!")
