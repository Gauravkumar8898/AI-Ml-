import tensorflow as tf


class CatsDogsModel:

    @staticmethod
    def create_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid',
                                         activation='relu', input_shape=(256, 256, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid',
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='valid',
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # model compile
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    @staticmethod
    def fit_model(model, train_dataset, test_dataset):
        model.fit(train_dataset, validation_data=test_dataset, epochs=2)
        return model

    @staticmethod
    def model_runner(train_dataset, test_dataset):
        obj = CatsDogsModel()
        model = obj.create_model()
        print(test_dataset)
        model = obj.fit_model(model, train_dataset, test_dataset)

        print(model.summary())
