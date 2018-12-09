import glob
import os

from TrafficLightDetector import TrafficLightDetector
from utilities import mkdir, numpyImage2PILImage, loadNumpyImage


class TrafficLightExtractor:

    def __init__(self, trafficLightDetector):
        self.trafficLightDetector = trafficLightDetector

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
        numpyImage = loadNumpyImage(imagePath)
        trafficLightNumpyImages = self.trafficLightDetector.detectTrafficLightsWithinNumpyImage(numpyImage)
        self.saveTrafficLights(imagePath, trafficLightNumpyImages, dst)

    def saveTrafficLights(self, filename, trafficLightNumpyImages, dst):
        for i, trafficLightNumpyImage in enumerate(trafficLightNumpyImages):
            self.saveTrafficLight(trafficLightNumpyImage, self.createFileName(dst, filename, i + 1))

    def createFileName(self, dst, filename, i):
        return dst + '/' + self.getNumberedFileName(filename, i)

    def getNumberedFileName(self, filename, i):
        root, extension = os.path.splitext(os.path.basename(filename))
        return root + '_' + str(i) + extension

    def saveTrafficLight(self, numpyImage, filename):
        numpyImage2PILImage(numpyImage).save(filename)


if __name__ == '__main__':
    trafficLightExtractor = TrafficLightExtractor(
        TrafficLightDetector('data/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'))
    trafficLightExtractor.extractAndSaveTrafficLights(srcDir='data/simulator_images', dstDir='data/trafficlight_images')
