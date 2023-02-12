from abc import ABC, abstractmethod
from tensorflow import keras


class Model(ABC):
    @abstractmethod
    def build_model(self, input_shape, nb_classes):
        pass


class SampleCNN(Model):
    def __init__(self, n_filters=32) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'SampleCNN'

    def build_model(self, input_shape, nb_classes):
        model = keras.models.Sequential([
            keras.layers.Conv2D(self.n_filters, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(self.n_filters * 2, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal'),
            keras.layers.Dense(nb_classes)
        ])
        return model

class LpdCNNa(Model):
    def __init__(self, n_filters=32) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'LpdCNNa'

    def build_model(self, input_shape, nb_classes):
        layers = [
            keras.layers.Conv2D(16, (4, 4), strides=(2, 2), padding='same', activation='relu', input_shape=(28, 28, 1), kernel_initializer='he_uniform'),
            keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_uniform')
        ]
        layers.append(keras.layers.Flatten())
        layers.append(keras.layers.Dense(100, activation='relu'))
        layers.append(keras.layers.Dense(10))
        return keras.models.Sequential(layers)
