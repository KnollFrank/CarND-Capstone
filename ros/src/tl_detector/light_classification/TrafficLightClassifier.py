from enum import Enum


class TrafficLight(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3


class TrafficLightClassifier:

    def classifyTrafficLights(self, image):
        return [TrafficLight.RED, TrafficLight.RED, TrafficLight.RED]
