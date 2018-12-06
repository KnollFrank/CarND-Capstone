from enum import Enum

from TrafficLightProvider import TrafficLightProvider
from PIL import Image
from keras.models import load_model

class TrafficLight(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3


class TrafficLightClassifier:

    def classifyTrafficLights(self, image):
        output_dict = TrafficLightProvider().detectTrafficLightsWithin2(image)
        trafficLights = []
        self.saveTrafficLights(image, output_dict['detection_boxes'], output_dict['detection_classes'], trafficLights)
        return trafficLights

    def saveTrafficLights(self, image, boxes, classes, trafficLights):
        for i, box in enumerate(boxes):
            self.saveTrafficLight(image, box, classes[i], trafficLights)

    def saveTrafficLight(self, image, box, clazz, trafficLights):
        # TODO: isTrafficLight() muß an höherer Stelle im Callgraph ausgeführt werden.
        if self.isTrafficLight(clazz):
            trafficLightImage = self.extractTrafficLight(box, image)
            color = self.detectColor(trafficLightImage)
            trafficLights.append(color)

    def isTrafficLight(self, clazz):
        return clazz == 10

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

    def detectColor(self, image):
        # model = load_model('/home/frankknoll/udacity/SDCND/CarND-Capstone/ros/src/tl_detector/light_classification/model.h5')
        return TrafficLight.RED
