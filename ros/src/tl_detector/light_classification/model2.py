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
bottleneck_features_train_file = 'bottleneck_features_train.npy'
bottleneck_features_validation_file = 'bottleneck_features_validation.npy'
bottleneck_features_train_labels = 'bottleneck_features_train_labels.npy'
bottleneck_features_validation_labels = 'bottleneck_features_validation_labels.npy'
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


def create_initialized_top_model_on_top_of_base_model():
    base_model = create_base_model()

    top_model = create_top_model(base_model.output_shape[1:])
    top_model.load_weights(top_model_weights_path)

    model = Sequential()
    model.add(base_model)
    model.add(top_model)

    return model


# see https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
def save_bottleneck_features():
    # build the network
    model = create_base_model()

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    # TODO: DRY with code near end of this function
    print('test data:')
    generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False,
        subset='training')
    bottleneck_features_train = model.predict_generator(generator, generator.n // generator.batch_size)
    np.save(open(bottleneck_features_train_file, 'wb'), bottleneck_features_train)
    np.save(open(bottleneck_features_train_labels, 'wb'), generator.classes)

    print('validation data:')
    generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False,
        subset='validation')
    bottleneck_features_validation = model.predict_generator(generator, generator.n // generator.batch_size)
    np.save(open(bottleneck_features_validation_file, 'wb'), bottleneck_features_validation)
    np.save(open(bottleneck_features_validation_labels, 'wb'), generator.classes)


def train_top_model():
    train_data = np.load(open(bottleneck_features_train_file, 'rb'))
    train_labels = np.load(open(bottleneck_features_train_labels, 'rb'))
    train_labels = np_utils.to_categorical(train_labels, num_classes)

    validation_data = np.load(open(bottleneck_features_validation_file, 'rb'))
    validation_labels = np.load(open(bottleneck_features_validation_labels, 'rb'))
    validation_labels = np_utils.to_categorical(validation_labels, num_classes)

    model = create_top_model(train_data.shape[1:])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data,
              train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    # callbacks=[ModelCheckpoint(filepath=top_model_weights_path, verbose=1, save_best_only=True)])
    model.save_weights(top_model_weights_path)


save_bottleneck_features()
train_top_model()

model = create_initialized_top_model_on_top_of_base_model()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.save(modelFile)
