import numpy as np
import tensorflow as tf
from PIL import Image
from enum import Enum
from keras.models import load_model
from keras.preprocessing import image

from helper import numpyImage2PILImage


class TrafficLightColor(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3


class TrafficLightClassifier:

    def __init__(self, modelFile, trafficLightDetector):
        self.model = load_model(modelFile)
        # see: https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
        self.graph = tf.get_default_graph()
        self.trafficLightDetector = trafficLightDetector

    def classifyTrafficLights(self, numpyImage):
        trafficLightNumpyImages = self.trafficLightDetector.detectTrafficLightsWithinNumpyImage(numpyImage)
        return self.detectTrafficLightColors(trafficLightNumpyImages)

    def detectTrafficLightColors(self, trafficLightNumpyImages):
        return [self.detectTrafficLightColor(trafficLightNumpyImage) for trafficLightNumpyImage in
                trafficLightNumpyImages]

    def detectTrafficLightColor(self, numpyImage):
        # TODO: refactor
        img_height, img_width = 120, 50
        PILImage = numpyImage2PILImage(numpyImage)
        PILImage = PILImage.resize((img_width, img_height), Image.ANTIALIAS)
        x = image.img_to_array(PILImage)
        x = np.expand_dims(x, axis=0)
        labels = {0: TrafficLightColor.GREEN, 1: TrafficLightColor.RED, 3: TrafficLightColor.YELLOW}
        with self.graph.as_default():
            prediction = self.model.predict(x)
        return labels[np.argmax(prediction)]
