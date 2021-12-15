import cv2
import os
import glob
from numpy.core.records import array
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from tensorflow.keras.utils import to_categorical
from time import time

import mediapipe as mp
from training.face_mesh import face_mesh

class preprocessor():
    def __init__(self, config) :
        self.face_mesh = face_mesh()

        # przypisanie źródeł danych
        self.src_dir = config["training_data"]         
        self.test_dir = config["test_data"]
        self.validation_src = config["validation_data"]
        self.dirs_labels = config["dirs_labels"]
        self.dimmensions = config["dimmensions"]
        self.type = config["type"]

    def check_for_valid_shape(self, images):
        for image in images:
            h, w, c = image.shape
            if h == 0 or w == 0 or c == 0: # każde zdjęcie musi posiadać niezerowy każdy z wymiarów
                return False
        return True

    def preprocess_image(self, image):
        preprocessed = np.zeros((1,1,3), dtype="uint8")
        preprocessed, array = self.face_mesh.cropp_by_eye_landmarks(image)
        if self.check_for_valid_shape(preprocessed): # sprawdzenie czy zdjęcia są odpowiednie
            # Konwersja skali, zwiększenie wartości alfa prowadzi do zwiększenia kontrastu zdjęcia - siatkówka lepiej widoczna
            # Zdjęcia są konwertowane do odpowiedniego rozmiaru wejściowego dla sieci
            return ([cv2.convertScaleAbs(cv2.resize(preprocessed[0], (self.dimmensions[0],self.dimmensions[1])), alpha=1.5),\
                    cv2.resize(preprocessed[1], (self.dimmensions[0],self.dimmensions[1]))], array)
        else :
            return None
        

    def load_and_preprocess_image(self, path):
        image = cv2.imread(path) # załadowanie zdjęcia ze ścieżki
        return self.preprocess_image(image)
        

    def load_data(self, src_dirs, dirs_labels):
        X = []
        Y = []

        files = {}
        for label in dirs_labels: # dla każdej etykiety
            for src_dir in src_dirs: # dla każdego folderu źródłowego
                class_folder_path = os.path.join(src_dir, label) # Wygeneruj ścieżkę do plików
                # Zaciągnij listę plików w folderze które kończą się na unique.jpg
                files[dirs_labels[label]] = glob.glob(class_folder_path + '/*unique.jpg', recursive=True) 
                for file in files[dirs_labels[label]]: # Dla każdego pliku z folderu
                    print("loading file ", file, " with label ", dirs_labels[label])
                    new_data = self.load_and_preprocess_image(file) # załaduj i obrób dane 
                    if new_data is not None: # sprawdź czy nie jest puste i załaduj do listy
                        X.append(new_data)
                        Y.append(dirs_labels[label])
        return X, Y

    def split_validation_set(self, X, y, test_size=0.2):
        print('Splitting the data...')

        random_state = int(time())
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=random_state)

        return X_train, X_test, y_train, y_test

    def split_images_from_arrays(self, data):
        images = []
        arrays = []
        for data_entry in data:
            images.append(data_entry[0])
            arrays.append(data_entry[1])
        return images, arrays

    def prepare_data(self):
        print("*********************** PREPARE TRAINING DATA *************************")
        
        # Załadowanie i obróbka wstępna zdjęć
        X,Y = self.load_data(self.src_dir, self.dirs_labels)
        Y = np.array(Y)
        X_val, Y_val = self.load_data(self.validation_src, self.dirs_labels)
        Y_val = np.array(Y_val)
        X_test, Y_test = self.load_data(self.test_dir, self.dirs_labels)
        Y_test = np.array(Y_test)
        
        # podział danych zbioru źródłowego
        X_train, x_val_splitted, y_train, y_val_splitted = self.split_validation_set(X, Y, 0.35) 

        # podział danych zbioru źródłowego
        X_test_splitted, x_val_splitted, y_test_splitted, y_val_splitted = self.split_validation_set(x_val_splitted, y_val_splitted, 0.5) 

        # X_val, Y_val = shuffle(X_val, Y_val, random_state=0)
        X_test, Y_test = shuffle(X_test, Y_test, random_state=0)
        
        # rozdział zdjęć od wartości skalarnych
        X_train_images, X_train_array = self.split_images_from_arrays(X_train)
        X_val_images, X_val_array = self.split_images_from_arrays(x_val_splitted)
        X_test_images, X_test_array = self.split_images_from_arrays(X_test_splitted)
             
        # przypisanie wartości do ciała obiektu
        self.X_train_images = np.array(X_train_images)/255
        self.X_validation_images = np.array(X_val_images)/255
        self.X_test_images = np.array(X_test_images)/255

        self.X_train_array = np.array(X_train_array)
        self.X_validation_array = np.array(X_val_array)
        self.X_test_array = np.array(X_test_array)

        self.y_train = np.array(y_train)
        self.y_validation = np.array(y_val_splitted)
        self.y_test = np.array(y_test_splitted)
        

        del X, Y, X_val, Y_val    

        print("Size of training set is ", self.X_train_images.shape)
        print("Size of test set is ", self.X_test_images.shape)
        print("Size of validation set is ", self.X_validation_images.shape)