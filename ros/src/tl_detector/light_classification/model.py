import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from preprocess import get_images_and_measurements, get_input_shape
from squeezenet import SqueezeNet

def get_X_train_y_train():
    images, measurements = get_images_and_measurements()
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train


def create_model_simple():
    model = Sequential()
    model.add(Flatten(input_shape=get_input_shape()))
    model.add(Dense(5, activation="softmax"))
    return model


def create_model_LeNet():
    model = Sequential()
    model.add(Lambda(lambda image: image / 255.0 - 0.5, input_shape=get_input_shape()))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    # TODO: generalize number of classes in all places
    model.add(Dense(5, activation="softmax"))
    return model


def create_model_Nvidia():
    model = Sequential()
    model.add(Lambda(lambda image: image / 255.0 - 0.5, input_shape=get_input_shape()))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(5, activation="softmax"))
    return model


def train(model, X, y, save_model_2_file):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_object = model.fit(X,
                               y,
                               validation_split=0.2,
                               shuffle=True,
                               callbacks=[ModelCheckpoint(filepath=save_model_2_file, verbose=1, save_best_only=True)],
                               epochs=5,
                               verbose=1)
    return history_object

def channels_last_2_channels_first(image):
    return np.moveaxis(image, -1, 0)

if __name__ == '__main__':
    X_train, y_train = get_X_train_y_train()
    (height, width, channels) = get_input_shape()
    # SqueezeNet braucht channels_first
    model = SqueezeNet(nb_classes = 5, inputs = (channels, height, width))
    X_train_channels_first = np.array(list(map(channels_last_2_channels_first, X_train)))
    train(model, X_train_channels_first, y_train, save_model_2_file='model_squeezenet.h5')
