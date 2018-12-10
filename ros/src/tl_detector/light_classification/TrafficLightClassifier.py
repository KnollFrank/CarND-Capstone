class TrafficLightClassifier:

    def __init__(self, trafficLightDetector, trafficLightColorClassifier):
        self.trafficLightDetector = trafficLightDetector
        self.trafficLightColorClassifier = trafficLightColorClassifier

    def classifyTrafficLights(self, numpyImage):
        trafficLightDescriptions = self.trafficLightDetector.detectTrafficLightsWithinNumpyImage(numpyImage)
        return self.detectTrafficLightColors(trafficLightDescriptions)

    def detectTrafficLightColors(self, trafficLightDescriptions):
        return map(self.detectTrafficLightColor, trafficLightDescriptions)

    def detectTrafficLightColor(self, trafficLightDescription):
        return self.trafficLightColorClassifier.detectTrafficLightColor(trafficLightDescription.trafficLightNumpyImage)
