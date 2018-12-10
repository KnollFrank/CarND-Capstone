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

class TrafficLightColorClassifierFactory:

    def __init__(self, epochs, modelFile):
        self.epochs = epochs
        self.modelFile = modelFile

        self.x_train_file = 'x_train.npy'
        self.y_train_file = 'y_train.npy'
        self.x_validation_file = 'x_validation.npy'
        self.y_validation_file = 'y_validation.npy'

        self.top_model_weights_file = 'top_model_weights.h5'

        self.batch_size = 16
        self.num_classes = 3

    def createAndSaveClassifier(self):
        self.save_bottleneck_features()
        self.train_and_save_top_model()
        self.create_and_save_initialized_top_model_on_top_of_base_model()

    def create_base_model(self):
        return applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        # return SqueezeNet(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    def create_top_model(self, input_shape):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='sigmoid'))
        return model

    # see https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
    def save_bottleneck_features(self):
        model = self.create_base_model()

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

        print('test data:')
        save_bottleneck_features('training', self.x_train_file, self.y_train_file)

        print('validation data:')
        save_bottleneck_features('validation', self.x_validation_file, self.y_validation_file)

    def train_and_save_top_model(self):
        x_train = np.load(open(self.x_train_file, 'rb'))
        y_train = np.load(open(self.y_train_file, 'rb'))
        y_train = np_utils.to_categorical(y_train, self.num_classes)

        x_validation = np.load(open(self.x_validation_file, 'rb'))
        y_validation = np.load(open(self.y_validation_file, 'rb'))
        y_validation = np_utils.to_categorical(y_validation, self.num_classes)

        model = self.create_top_model(x_train.shape[1:])
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train,
                  y_train,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  validation_data=(x_validation, y_validation))
        # callbacks=[ModelCheckpoint(filepath=top_model_weights_path, verbose=1, save_best_only=True)])
        model.save_weights(self.top_model_weights_file)

    def create_initialized_top_model_on_top_of_base_model(self):
        base_model = self.create_base_model()

        top_model = self.create_top_model(base_model.output_shape[1:])
        top_model.load_weights(self.top_model_weights_file)

        model = Sequential()
        model.add(base_model)
        model.add(top_model)

        return model

    def create_and_save_initialized_top_model_on_top_of_base_model(self):
        model = self.create_initialized_top_model_on_top_of_base_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])
        model.save(modelFile)


if __name__ == '__main__':
    classifierFactory = TrafficLightColorClassifierFactory(epochs=50, modelFile='model.h5')
    classifierFactory.createAndSaveClassifier()
