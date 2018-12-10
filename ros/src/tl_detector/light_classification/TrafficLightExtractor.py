import glob
import os

from TrafficLightDetector import TrafficLightDetector
from utilities import mkdir, numpyImage2PILImage, loadNumpyImage, PILImage2numpyImage, resizePILImage, loadPILImage


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
        numpyImage = PILImage2numpyImage(resizePILImage(PILImage, width=width / 4, height=height / 4))
        trafficLightDescriptions = self.trafficLightDetector.detectTrafficLightsWithinNumpyImage(numpyImage)
        trafficLightNumpyImages = map(lambda trafficLightDescription: trafficLightDescription.trafficLightNumpyImage,
                                      trafficLightDescriptions)
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
        print 'saving: ' + filename
        numpyImage2PILImage(numpyImage).save(filename)


if __name__ == '__main__':
    trafficLightExtractor = TrafficLightExtractor(
        TrafficLightDetector(
            'data/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb'))
    trafficLightExtractor.extractAndSaveTrafficLights(srcDir='data/simulator_images',
                                                      dstDir='data/trafficlight_small_images')
