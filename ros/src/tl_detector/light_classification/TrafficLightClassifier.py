import numpy as np
import tensorflow as tf
from PIL import Image
from enum import Enum
from keras.models import load_model
from keras.preprocessing import image


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
        boxes = self.trafficLightDetector.detectTrafficLightsWithinNumpyImage(numpyImage)
        trafficLights = []
        self.saveTrafficLights(numpyImage, boxes, trafficLights)
        return trafficLights

    # TODO: refactor
    def saveTrafficLights(self, image, boxes, trafficLights):
        for box in boxes:
            self.saveTrafficLight(image, box, trafficLights)

    def saveTrafficLight(self, image, box, trafficLights):
        trafficLightImage = self.extractTrafficLight(box, image)
        color = self.detectColor(trafficLightImage)
        trafficLights.append(color)

    def extractTrafficLight(self, box, image):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        return image.crop(self.adaptBox2Image(box, image))

    def adaptBox2Image(self, box, image):
        width, height = image.size
        left = box[1] * width
        upper = box[0] * height
        right = box[3] * width
        lower = box[2] * height
        return map(int, (left, upper, right, lower))

    def detectColor(self, img):
        img_height, img_width = 120, 50
        img = img.resize((img_width, img_height), Image.ANTIALIAS)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        labels = {0: TrafficLight.GREEN, 1: TrafficLight.RED, 3: TrafficLight.YELLOW}
        with self.graph.as_default():
            prediction = self.model.predict(x)
        return labels[np.argmax(prediction)]
