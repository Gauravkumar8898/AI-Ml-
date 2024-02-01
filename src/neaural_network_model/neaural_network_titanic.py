import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class TitanicModel:

    @staticmethod
    def create_model():
        # Create a sequential model
        model = tf.keras.Sequential([
            # First dense layer with 23 units and ReLU activation function
            tf.keras.layers.Dense(23, activation='relu'),
            # Dropout layer with a dropout rate of 20%
            tf.keras.layers.Dropout(0.2),
            # Second dense layer with 15 units and ReLU activation function
            tf.keras.layers.Dense(15, activation='relu'),
            # Batch normalization layer
            tf.keras.layers.BatchNormalization(),
            # Third dense layer with 10 units and linear activation function
            tf.keras.layers.Dense(10, activation='linear'),
            # Output layer with 1 unit and sigmoid activation function
            tf.keras.layers.Dense(1, activation='sigmoid')])

        # compile the model
        model.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        return model

    @staticmethod
    def plot_tran_validation(training):
        """
        Plot training and validation accuracy over epochs.
        Parameters:
            training: History object returned by the model.fit() function.
        """
        plt.plot(training.history['accuracy'])
        plt.plot(training.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    @staticmethod
    def work_flow_neural_network(x_train_scaled, y_train_scaled, x_test, y_test):
        obj = TitanicModel()
        model = obj.create_model()
        training = model.fit(x_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
        obj.plot_tran_validation(training)
        val_acc = np.mean(training.history['val_accuracy'])
        print("\n%s: %.2f%%" % ('val_acc', val_acc * 100))
        print(model.evaluate(x_test, y_test))
