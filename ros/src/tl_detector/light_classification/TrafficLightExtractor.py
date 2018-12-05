from utilities import mkdir

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import glob

sys.path.append('/home/frankknoll/udacity/SDCND/models/research')
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util


# adapted from https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
class TrafficLightExtractor:

    def __init__(self):
        if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
            raise ImportError('Please upgrade your TensorFlow installation from ' + str(
                StrictVersion(tf.__version__)) + ' to v1.9.* or later!')

        self.PATH_TO_FROZEN_GRAPH = '/home/frankknoll/udacity/SDCND/models/research/object_detection/rfcn_resnet101_coco_2018_01_28//frozen_inference_graph.pb'
        self.detection_graph = self.load_model()

    def load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def extractAndSaveTrafficLights(self, srcDir, dstDir):
        self.extractAndSaveTrafficLightsForColor('red', srcDir, dstDir)
        self.extractAndSaveTrafficLightsForColor('yellow', srcDir, dstDir)
        self.extractAndSaveTrafficLightsForColor('green', srcDir, dstDir)

    def extractAndSaveTrafficLightsForColor(self, color, srcDir, dstDir):
        mkdir(self.getDir4Color(dstDir, color))
        self.detectAndSaveTrafficLights(glob.glob(self.getDir4Color(srcDir, color) + '/*'),
                                        self.getDir4Color(dstDir, color))

    def getDir4Color(self, dir, color):
        return dir + '/' + color

    def detectAndSaveTrafficLights(self, imagePaths, dst):
        for imagePath in imagePaths:
            self.detectAndSaveTrafficLightsWithinImage(imagePath, dst)

    def detectAndSaveTrafficLightsWithinImage(self, imagePath, dst):
        image = Image.open(imagePath)
        output_dict = self.detectTrafficLightsWithin(image)
        self.saveTrafficLights(image, output_dict['detection_boxes'], output_dict['detection_classes'], dst)

    def detectTrafficLightsWithin(self, image):
        return self.run_inference_for_single_image(self.load_image_into_numpy_array(image))

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def saveTrafficLights(self, image, boxes, classes, dst):
        for i, box in enumerate(boxes):
            self.saveTrafficLight(image, box, classes[i], self.createFileName(dst, image, i + 1))

    def createFileName(self, dst, image, i):
        return dst + '/' + self.getNumberedFileName(image.filename, i)

    def getNumberedFileName(self, filename, i):
        root, extension = os.path.splitext(os.path.basename(filename))
        return root + '_' + str(i) + extension

    def saveTrafficLight(self, image, box, clazz, filename):
        # TODO: isTrafficLight() muß an höherer Stelle im Callgraph ausgeführt werden.
        if self.isTrafficLight(clazz):
            trafficLight = self.extractTrafficLight(box, image)
            trafficLight.save(filename)

    def isTrafficLight(self, clazz):
        return clazz == 10

    def extractTrafficLight(self, box, image):
        return image.crop(self.adaptBox2Image(box, image))

    def adaptBox2Image(self, box, image):
        width, height = image.size
        left = box[1] * width
        upper = box[0] * height
        right = box[3] * width
        lower = box[2] * height
        return map(int, (left, upper, right, lower))
