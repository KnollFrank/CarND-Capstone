import numpy as np
import tensorflow as tf
from PIL import Image
from enum import Enum
from keras.models import load_model
from keras.preprocessing import image

from helper import numpyImage2PILImage


class TrafficLight(Enum):
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
        trafficLights = []
        self.saveTrafficLights(numpyImage, trafficLightNumpyImages, trafficLights)
        return trafficLights

    # TODO: refactor
    def saveTrafficLights(self, numpyImage, trafficLightNumpyImages, trafficLights):
        for trafficLightNumpyImage in trafficLightNumpyImages:
            self.saveTrafficLight(numpyImage, trafficLightNumpyImage, trafficLights)

    def saveTrafficLight(self, numpyImage, trafficLightNumpyImage, trafficLights):
        # trafficLightPILImage = self.extractTrafficLight(trafficLightNumpyImage, numpyImage)
        color = self.detectColor(trafficLightNumpyImage)
        trafficLights.append(color)

    def extractTrafficLight(self, trafficLightNumpyImage, PILImage):
        PILImage = Image.fromarray(PILImage.astype('uint8'), 'RGB')
        return PILImage.crop(self.adaptBox2Image(trafficLightNumpyImage, PILImage))

    def adaptBox2Image(self, trafficLightNumpyImage, numpyImage):
        width, height = numpyImage.size
        left = trafficLightNumpyImage[1] * width
        upper = trafficLightNumpyImage[0] * height
        right = trafficLightNumpyImage[3] * width
        lower = trafficLightNumpyImage[2] * height
        return map(int, (left, upper, right, lower))

    def detectColor(self, trafficLightNumpyImage):
        img_height, img_width = 120, 50
        trafficLightPILImage = numpyImage2PILImage(trafficLightNumpyImage)
        trafficLightPILImage = trafficLightPILImage.resize((img_width, img_height), Image.ANTIALIAS)
        x = image.img_to_array(trafficLightPILImage)
        x = np.expand_dims(x, axis=0)
        labels = {0: TrafficLight.GREEN, 1: TrafficLight.RED, 3: TrafficLight.YELLOW}
        with self.graph.as_default():
            prediction = self.model.predict(x)
        return labels[np.argmax(prediction)]
