import tensorflow as tf
from src.utils.constant import train_dogs_cats_data_path, test_dogs_cats_data_path


class Preprocessing:

    @staticmethod
    def load_dataset(dataset_path):
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=dataset_path,
            labels='inferred',
            label_mode='int',
            batch_size=32,
            image_size=(256, 256))
        return dataset

    @staticmethod
    def scaled_images(images, label):
        images = tf.cast(images / 255, tf.float32)
        return images, label

    @staticmethod
    def runner_preprocessing():
        obj = Preprocessing()
        train_dataset = obj.load_dataset(train_dogs_cats_data_path)
        test_dataset = obj.load_dataset(test_dogs_cats_data_path)
        train_dataset = train_dataset.map(obj.scaled_images)
        test_dataset = test_dataset.map(obj.scaled_images)
        return train_dataset, test_dataset

