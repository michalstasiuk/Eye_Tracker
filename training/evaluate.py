import tensorflow as tf
import json
import sys
from time import time
import numpy as np
import cv2
from sklearn.metrics import  mean_absolute_percentage_error, explained_variance_score


from preprocessing import preprocessor
from model import eye_tracker

class evaluation:
    def __init__(self, model, config) :
        self.model = model
        self.config = config
        self.labels = config["dirs_labels"]
        self.dimmensions = config["dimmensions"]
        self.type = config["type"]
        self.preprocessor = preprocessor(self.config)

    def evaluate_model(self):
        self.preprocessor.prepare_data()

        # wybranie potrzebnych na czas treningu i/lub ewaluacji danych
        X_images = self.preprocessor.X_validation_images
        X_array = self.preprocessor.X_validation_array
        Y = self.preprocessor.y_validation

        # transpozycja pierwszych dwóch osi z formatu (Ilość, Rodzaj_obrazu, Wymiary...)
        # na format (Rodzaj_obrazu, Ilość, Wymiary...)
        X_images = np.transpose(X_images,(1,0,2,3,4)) 

        # inicjalizacji ciągów i słowników przetrzymujących wyniki predykcji
        y_pred = []
        y_pred_dict = {
            "0.1" : [0.0],
            "0.2" : [0.0],
            "0.3" : [0.0],
            "0.4" : [0.0],
            "0.5" : [0.0],
            "0.6" : [0.0],
            "0.7" : [0.0],
            "0.8" : [0.0],
            "0.9" : [0.0]
        }

        percentage_error_per_angle = {
            "0.1" : [0.0],
            "0.2" : [0.0],
            "0.3" : [0.0],
            "0.4" : [0.0],
            "0.5" : [0.0],
            "0.6" : [0.0],
            "0.7" : [0.0],
            "0.8" : [0.0],
            "0.9" : [0.0]
        }

        percentage_error_per_angle_modfied = {
            "0.1" : [0.0],
            "0.2" : [0.0],
            "0.3" : [0.0],
            "0.4" : [0.0],
            "0.5" : [0.0],
            "0.6" : [0.0],
            "0.7" : [0.0],
            "0.8" : [0.0],
            "0.9" : [0.0]
        }

        for i in range(len(X_array)): # dla każdego elementu wykonaj predykcję
            pred = self.model.predict([X_images[0][i].reshape(-1,64,64,3),
                                       X_images[1][i].reshape(-1,64,64,3),
                                       X_array[i].reshape(-1,6)])[0][0]
            y_pred_dict[str(Y[i])].append(pred) # do tablicy w kluczu Y[i] dodaj nową wartość
            calculated_error = np.abs(pred-Y[i])/Y[i]
            calculated_error_modified = np.abs(pred-Y[i]) * 10
            percentage_error_per_angle[str(Y[i])].append(calculated_error)
            percentage_error_per_angle_modfied[str(Y[i])].append(calculated_error_modified)
            y_pred.append(pred)
            print("added new prediction: ", i)

        self.make_clusterization(y_pred_dict) # wykonaj klasteryzację za pomocą danych słownikowych

        y_pred = np.array(y_pred) # przekonwertuj na tablicę numpy

        for i in range(len(Y)): # dla każdej etykiety prawdziwej wypisz wynik prawdziwy oraz predykowany
            print(Y[i], ": ", y_pred[i]) 

        percentage_error = mean_absolute_percentage_error(Y, y_pred) # oblicz średni błąd procentowy
        explained_variance = explained_variance_score(Y, y_pred) # oblicz wyjaśnioną wariancję

        for i in range(9):
            key = "0." + str(i+1)
            error_in_angle = sorted(percentage_error_per_angle[key])
            best_95 = error_in_angle[:int(len(error_in_angle)*0.95)]
            worst_5 = error_in_angle[-int(len(error_in_angle)*0.05):]
            print("average for ", str(i+1), ": ", np.average(error_in_angle), " worst 5%: ", np.average(worst_5), " best 95%: ", np.average(best_95))

        for i in range(9):
            key = "0." + str(i+1)
            error_in_angle = sorted(percentage_error_per_angle_modfied[key])
            best_95 = error_in_angle[:int(len(error_in_angle)*0.95)]
            worst_5 = error_in_angle[-int(len(error_in_angle)*0.05):]
            print("modified average for ", str(i+1), ": ", np.average(error_in_angle), " worst 5%: ", np.average(worst_5), " best 95%: ", np.average(best_95))
            


        # ewaluacja modelu na zbiorze testowym z domyślną metryką MSE
        results = self.model.evaluate([X_images[0], X_images[1], X_array], Y, batch_size = 32) 
        print("test loss", results)
        print("percentage_error:", percentage_error)
        print("explained_variance:", explained_variance)
        
        return results
    
    def make_clusterization(self, y_pred_dict):
        img = np.zeros([550,1100,3],dtype=np.uint8)
        img.fill(255)
        cv2.line(img, (50, 500), (1050, 500), (0,0,0), 1)

        cv2.line(img, (150+0*100, 500), (150+0*100, 50), (255,0,0), 1,)
        cv2.line(img, (150+1*100, 500), (150+1*100, 50), (255,255,0), 1,)
        cv2.line(img, (150+2*100, 500), (150+2*100, 50), (255,0,255), 1,)
        cv2.line(img, (150+3*100, 500), (150+3*100, 50), (0,255,0), 1,)
        cv2.line(img, (150+4*100, 500), (150+4*100, 50), (0,0,255), 1,)
        cv2.line(img, (150+5*100, 500), (150+5*100, 50), (0,255,255), 1,)
        cv2.line(img, (150+6*100, 500), (150+6*100, 50), (128,192,64), 1,)
        cv2.line(img, (150+7*100, 500), (150+7*100, 50), (64,128,192), 1,)
        cv2.line(img, (150+8*100, 500), (150+8*100, 50), (192,64,128), 1,)

        for i in range(len(y_pred_dict["0.1"])):
            cv2.circle(img, (int(50+1000*y_pred_dict["0.1"][i]), int(480 - 430/len(y_pred_dict["0.1"])*i)), 1, (255,0,0), 2)
        for i in range(len(y_pred_dict["0.2"])):
            cv2.circle(img, (int(50+1000*y_pred_dict["0.2"][i]), int(480 - 430/len(y_pred_dict["0.2"])*i)), 1, (255,255,0), 2)
        for i in range(len(y_pred_dict["0.3"])):
            cv2.circle(img, (int(50+1000*y_pred_dict["0.3"][i]), int(480 - 430/len(y_pred_dict["0.3"])*i)), 1, (255,0,255), 2)
        for i in range(len(y_pred_dict["0.4"])):
            cv2.circle(img, (int(50+1000*y_pred_dict["0.4"][i]), int(480 - 430/len(y_pred_dict["0.4"])*i)), 1, (0,255,0), 2)
        for i in range(len(y_pred_dict["0.5"])):
            cv2.circle(img, (int(50+1000*y_pred_dict["0.5"][i]), int(480 - 430/len(y_pred_dict["0.5"])*i)), 1, (0,0,255), 2)
        for i in range(len(y_pred_dict["0.6"])):
            cv2.circle(img, (int(50+1000*y_pred_dict["0.6"][i]), int(480 - 430/len(y_pred_dict["0.6"])*i)), 1, (0,255,255), 2)
        for i in range(len(y_pred_dict["0.7"])):
            cv2.circle(img, (int(50+1000*y_pred_dict["0.7"][i]), int(480 - 430/len(y_pred_dict["0.7"])*i)), 1, (128,192,64), 2)
        for i in range(len(y_pred_dict["0.8"])):
            cv2.circle(img, (int(50+1000*y_pred_dict["0.8"][i]), int(480 - 430/len(y_pred_dict["0.8"])*i)), 1, (64,128,192), 2)
        for i in range(len(y_pred_dict["0.9"])):
            cv2.circle(img, (int(50+1000*y_pred_dict["0.9"][i]), int(480 - 430/len(y_pred_dict["0.9"])*i)), 1, (192,64,128), 2)

        cv2.imwrite("clusterization.jpg", img)

    def show_results_on_video(self, video_path):
        if video_path == "0":
            cap = cv2.VideoCapture(int(video_path))
        else:
            cap = cv2.VideoCapture(video_path)
        print("Start video stream")
        with open("./evaluation.csv", "w+") as file:
            while cap.isOpened():
                success, image = cap.read()
                if success :
                    preprocessed = self.preprocessor.preprocess_image(image)
                    preprocessed = np.reshape(preprocessed,[1,224,224,3])
                    output = self.model.predict(preprocessed)[0]
                    output = np.array(output)
                    print(output)
                    max_value_index = np.argmax(output)
                    label = list(self.labels.keys())[max_value_index]
                    max_value = output[max_value_index]
                    
                    text = label + " with propability " + str(max_value)
                    print(text)
                    ratio=image.shape[1]/image.shape[0]
                    new_size = (int(1080*ratio),1080)
                    print(new_size)

                    image = cv2.resize(image, new_size)
                    cv2.putText(image, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                    cv2.imshow("image", image)
                    cv2.waitKey(33)
                else :
                    print("End of video stream, exiting")
                    exit()

if __name__ == "__main__":

    if len(sys.argv) < 2 :
        print("not provided configuration file, evaluating from test set")

    file = open(sys.argv[1])
    if len(sys.argv) == 3 :   
        video_path = sys.argv[2]
    data = json.load(file)

    model = eye_tracker()
    weights_path = "./release/model_weights_eye_tracker.h5"
    model.load_weights(weights_path)

    eval = evaluation(model, data)
    if len(sys.argv) == 3 :
        eval.show_results_on_video(video_path)
    else:
        eval.evaluate_model()