import cv2
import json
import numpy as np
import time

from inferencer import Inferencer
from training.preprocessing import preprocessor

class Eye_Tracker:
    def __init__(self, config):
        self.dimmensions = config["dimmensions"]
        model_path = config["model_path"]

        self.preprocessor = preprocessor(config)
        self.inferencer = Inferencer(model_path)

    def preprocess(self, image):
        start_preprocessing = time.time()

        data = self.preprocessor.preprocess_image(image)

        stop_preprocessing = time.time()
        print("TIME: PREPROCESSING: ", stop_preprocessing - start_preprocessing)

        return data

    def run(self, image):
        data = self.preprocess(image)
        images = np.array(data[0], dtype=np.float32)/255
        arrays = np.array(data[1], dtype=np.float32)
        output = self.inferencer.run(images, arrays)
        output = np.array(output)
        print(output)

        return output