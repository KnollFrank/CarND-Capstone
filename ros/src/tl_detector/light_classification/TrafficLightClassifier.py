class TrafficLightClassifier:

    def __init__(self, trafficLightDetector, trafficLightColorClassifier, minScore):
        self.trafficLightDetector = trafficLightDetector
        self.trafficLightColorClassifier = trafficLightColorClassifier
        self.minScore = minScore

    def classifyTrafficLights(self, numpyImage):
        trafficLightDescriptions = self.filterByMinScore(
            self.trafficLightDetector.detectTrafficLightsWithinNumpyImage(numpyImage))
        return self.detectTrafficLightColors(trafficLightDescriptions)

    # TODO: DRY with TrafficLightExtractor.filterByMinScore()
    def filterByMinScore(self, trafficLightDescriptions):
        return filter(lambda trafficLightDescription: trafficLightDescription.score >= self.minScore,
                      trafficLightDescriptions)

    def detectTrafficLightColors(self, trafficLightDescriptions):
        return map(self.detectTrafficLightColor, trafficLightDescriptions)

    def detectTrafficLightColor(self, trafficLightDescription):
        return self.trafficLightColorClassifier.detectTrafficLightColor(trafficLightDescription.trafficLightNumpyImage)
