import numpy as np
import cv2
import tensorflow as tf
import glob
import time


class Inferencer:
    
    def __init__(self, model_path) :
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run(self, images, array):
        #preprocessed_image = np.reshape(image, (1,)+image.shape)
        self.interpreter.set_tensor(self.input_details[0]['index'], images[0].reshape(-1,64,64,3))
        self.interpreter.set_tensor(self.input_details[1]['index'], images[1].reshape(-1,64,64,3))
        self.interpreter.set_tensor(self.input_details[2]['index'], array.reshape(-1,6))

        start_invoke = time.time()

        self.interpreter.invoke()

        stop_invoke = time.time()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        print("TIME: INFERENCE: ", stop_invoke - start_invoke)

        return output_data

    def get_predicted_class_with_propability(self, tensor):
        return np.argmax(tensor), tensor[np.argmax(tensor)]