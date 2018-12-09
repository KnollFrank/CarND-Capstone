from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import applications
# from squeezenet import SqueezeNet
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.utils import np_utils

# dimensions of our images.
img_height, img_width = 120, 50

train_data_dir = 'data/trafficlight_images'
# TODO: rename to top_model_weights_file
top_model_weights_path = 'bottleneck_fc_model.h5'
modelFile = 'model.h5'

num_classes = 3
epochs = 50
batch_size = 16


def create_base_model():
    return applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    # return SqueezeNet(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))


def create_top_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model


# see https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
def save_bottleneck_features():
    model = create_base_model()

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    def save_bottleneck_features(subset, x_file, y_file):
        generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=1,
            class_mode=None,
            shuffle=False,
            subset=subset)

        x_bottleneck = model.predict_generator(generator, generator.n // generator.batch_size)
        np.save(open(x_file, 'wb'), x_bottleneck)

        y_bottleneck = generator.classes
        np.save(open(y_file, 'wb'), y_bottleneck)

    x_train_file = 'x_train.npy'
    y_train_file = 'y_train.npy'

    x_validation_file = 'x_validation.npy'
    y_validation_file = 'y_validation.npy'

    print('test data:')
    save_bottleneck_features('training', x_train_file, y_train_file)

    print('validation data:')
    save_bottleneck_features('validation', x_validation_file, y_validation_file)

    return x_train_file, y_train_file, x_validation_file, y_validation_file


def train_top_model(x_train_file, y_train_file, x_validation_file, y_validation_file):
    x_train = np.load(open(x_train_file, 'rb'))
    y_train = np.load(open(y_train_file, 'rb'))
    y_train = np_utils.to_categorical(y_train, num_classes)

    x_validation = np.load(open(x_validation_file, 'rb'))
    y_validation = np.load(open(y_validation_file, 'rb'))
    y_validation = np_utils.to_categorical(y_validation, num_classes)

    model = create_top_model(x_train.shape[1:])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train,
              y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_validation, y_validation))
    # callbacks=[ModelCheckpoint(filepath=top_model_weights_path, verbose=1, save_best_only=True)])
    model.save_weights(top_model_weights_path)


def create_initialized_top_model_on_top_of_base_model():
    base_model = create_base_model()

    top_model = create_top_model(base_model.output_shape[1:])
    top_model.load_weights(top_model_weights_path)

    model = Sequential()
    model.add(base_model)
    model.add(top_model)

    return model


def create_and_save_initialized_top_model_on_top_of_base_model():
    model = create_initialized_top_model_on_top_of_base_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    model.save(modelFile)


if __name__ == '__main__':
    x_train_file, y_train_file, x_validation_file, y_validation_file = save_bottleneck_features()
    train_top_model(x_train_file, y_train_file, x_validation_file, y_validation_file)
    create_and_save_initialized_top_model_on_top_of_base_model()
