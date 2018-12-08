from distutils.version import StrictVersion

import numpy as np
import tensorflow as tf


# adapted from https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
class TrafficLightDetector:

    def __init__(self, path2FrozenGraph):
        self.detection_graph = self.load_model(path2FrozenGraph)

    def load_model(self, path2FrozenGraph):
        if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
            raise ImportError('Please upgrade your TensorFlow installation from ' + str(
                StrictVersion(tf.__version__)) + ' to v1.9.* or later!')

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path2FrozenGraph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def detectTrafficLightsWithinNumpyImage(self, numpyImage):
        output_dict = self.run_inference_for_single_image(numpyImage)
        trafficLightNumpyImages = []
        for i, box in enumerate(output_dict['detection_boxes']):
            if self.isTrafficLight(output_dict['detection_classes'][i]):
                trafficLightNumpyImage = self.crop(numpyImage, box)
                trafficLightNumpyImages.append(trafficLightNumpyImage)

        return trafficLightNumpyImages

    def crop(self, numpyImage, fractionBox):
        left, lower, right, upper = self.adaptFractionBox2Image(fractionBox, numpyImage)
        imageNumpy = numpyImage[upper:lower + 1, left:right + 1, :]
        return imageNumpy

    def adaptFractionBox2Image(self, fractionBox, numpyImage):
        upperFraction, leftFraction, lowerFraction, rightFraction = fractionBox
        (height, width, _) = numpyImage.shape
        upper, left, lower, right = map(int, (
            upperFraction * height, leftFraction * width, lowerFraction * height, rightFraction * width))
        return left, lower, right, upper

    def isTrafficLight(self, clazz):
        return clazz == 10

    def run_inference_for_single_image(self, numpyImage):
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
                        detection_masks, detection_boxes, numpyImage.shape[0], numpyImage.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(numpyImage, 0)})

                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict
