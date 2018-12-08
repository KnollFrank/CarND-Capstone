from TrafficLightDetector import TrafficLightDetector
from helper import PILImage2numpyImage, numpyImage2PILImage
from utilities import mkdir

import os
import sys

from PIL import Image
import glob

# TODO: remove the following three lines:
sys.path.append('/home/frankknoll/udacity/SDCND/models/research')
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

class TrafficLightExtractor:

    def __init__(self, PATH_TO_FROZEN_GRAPH):
        self.trafficLightDetector = TrafficLightDetector(PATH_TO_FROZEN_GRAPH)

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
        PILImage = Image.open(imagePath)
        trafficLightNumpyImages = self.trafficLightDetector.detectTrafficLightsWithinNumpyImage(
            PILImage2numpyImage(PILImage))
        self.saveTrafficLights(PILImage, trafficLightNumpyImages, dst)

    def saveTrafficLights(self, PILImage, trafficLightNumpyImages, dst):
        for i, trafficLightNumpyImage in enumerate(trafficLightNumpyImages):
            self.saveTrafficLight(trafficLightNumpyImage, self.createFileName(dst, PILImage, i + 1))

    def createFileName(self, dst, image, i):
        return dst + '/' + self.getNumberedFileName(image.filename, i)

    def getNumberedFileName(self, filename, i):
        root, extension = os.path.splitext(os.path.basename(filename))
        return root + '_' + str(i) + extension

    def saveTrafficLight(self, numpyImage, filename):
        numpyImage2PILImage(numpyImage).save(filename)


if __name__ == '__main__':
    trafficLightExtractor = TrafficLightExtractor('data/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb')
    trafficLightExtractor.extractAndSaveTrafficLights(srcDir='data/simulator_images', dstDir='data/trafficlight_images')
