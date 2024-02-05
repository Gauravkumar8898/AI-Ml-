import tensorflow as tf
import matplotlib.pyplot as plt


def pre_trained_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(256, 256, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    return base_model


def create_model(cov_base):
    model = tf.keras.Sequential()
    cov_base.trainable = False
    model.add(cov_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def model_fit(model, train_ds, test_ds):
    history = model.fit(train_ds, epochs=1, validation_data=test_ds)
    return history


def evaluate_model(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def transfer_learning_runner(train_dataset, test_dataset):
    base_model = pre_trained_model()
    model = create_model(base_model)
    history = model_fit(model, train_dataset, test_dataset)
    evaluate_model(history)
