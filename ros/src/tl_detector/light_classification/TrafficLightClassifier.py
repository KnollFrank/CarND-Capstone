class TrafficLightClassifier:

    def __init__(self, trafficLightDetector, trafficLightColorClassifier):
        self.trafficLightDetector = trafficLightDetector
        self.trafficLightColorClassifier = trafficLightColorClassifier

    def classifyTrafficLights(self, numpyImage):
        trafficLightNumpyImages = self.trafficLightDetector.detectTrafficLightsWithinNumpyImage(numpyImage)
        return self.detectTrafficLightColors(trafficLightNumpyImages)

    def detectTrafficLightColors(self, trafficLightNumpyImages):
        return [self.trafficLightColorClassifier.detectTrafficLightColor(trafficLightNumpyImage) for
                trafficLightNumpyImage in
                trafficLightNumpyImages]
