from src.pipeline.titanic_pipeline import pipeline
from src.pipeline.mnist_pipeline import mnist_pipeline_


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    inputs = abs(int(input("Enter Choice for mnist = 1 and titanic = 2:")))
    if inputs == 1:
        pipeline()
    elif inputs == 2:
        mnist_pipeline_()
    else:
        print("Enter  1 or 2 only!")
