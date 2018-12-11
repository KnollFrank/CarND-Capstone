class TrafficLightHavingMinScoreDetector:

    def __init__(self, trafficLightDetector):
        self.trafficLightDetector = trafficLightDetector
        self.minScore = 0.6

    def detectTrafficLightsWithinNumpyImage(self, numpyImage):
        return self.filterByMinScore(self.trafficLightDetector.detectTrafficLightsWithinNumpyImage(numpyImage))

    def filterByMinScore(self, trafficLightDescriptions):
        return filter(lambda trafficLightDescription: trafficLightDescription.score >= self.minScore,
                      trafficLightDescriptions)
