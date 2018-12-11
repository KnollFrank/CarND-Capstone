import glob
import os

from TrafficLightDetector import TrafficLightDetector
from TrafficLightHavingMinScoreDetector import TrafficLightHavingMinScoreDetector
from utilities import mkdir, numpyImage2PILImage, PILImage2numpyImage, resizePILImage, loadPILImage

TRAFFIC_LIGHT_DETECTOR_NAME = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'


class TrafficLightExtractor:

    def __init__(self, trafficLightDetector):
        self.trafficLightDetector = trafficLightDetector

    # @profile
    def extractAndSaveTrafficLights(self, srcDir, dstDir):
        self.extractAndSaveTrafficLights4Color('red', srcDir, dstDir)
        self.extractAndSaveTrafficLights4Color('yellow', srcDir, dstDir)
        self.extractAndSaveTrafficLights4Color('green', srcDir, dstDir)

    def extractAndSaveTrafficLights4Color(self, color, srcDir, dstDir):
        mkdir(self.getDir4Color(dstDir, color))
        self.detectAndSaveTrafficLights(self.getFiles4Color(srcDir, color),
                                        self.getDir4Color(dstDir, color))

    def getFiles4Color(self, dir, color):
        return glob.glob(self.getDir4Color(dir, color) + '/*')

    def getDir4Color(self, dir, color):
        return dir + '/' + color

    def detectAndSaveTrafficLights(self, imagePaths, dst):
        for imagePath in imagePaths:
            self.detectAndSaveTrafficLightsWithinImage(imagePath, dst)

    def detectAndSaveTrafficLightsWithinImage(self, imagePath, dst):
        PILImage = loadPILImage(imagePath)
        width, height = PILImage.size
        numpyImage = PILImage2numpyImage(resizePILImage(PILImage, width=width, height=height))
        trafficLightDescriptions = self.trafficLightDetector.detectTrafficLightsWithinNumpyImage(numpyImage)
        self.saveTrafficLights(imagePath, trafficLightDescriptions, dst)

    def saveTrafficLights(self, filename, trafficLightDescriptions, dst):
        for i, trafficLightDescription in enumerate(trafficLightDescriptions):
            self.saveTrafficLight(trafficLightDescription.trafficLightNumpyImage,
                                  self.createFileName(dst, filename, i + 1, trafficLightDescription.score))

    def createFileName(self, dst, filename, i, score):
        return dst + '/' + self.getNumberedFileName(filename, i, score)

    def getNumberedFileName(self, filename, i, score):
        root, extension = os.path.splitext(os.path.basename(filename))
        return root + '_' + str(i) + '_' + str(score) + extension

    def saveTrafficLight(self, numpyImage, filename):
        print 'saving: ' + filename
        numpyImage2PILImage(numpyImage).save(filename)


if __name__ == '__main__':
    trafficLightExtractor = TrafficLightExtractor(
        TrafficLightHavingMinScoreDetector(
            TrafficLightDetector('data/' + TRAFFIC_LIGHT_DETECTOR_NAME + '/frozen_inference_graph.pb'),
            minScore=0.5))
    trafficLightExtractor.extractAndSaveTrafficLights(srcDir='data/simulator_images',
                                                      dstDir='data/trafficlight_small_images')
