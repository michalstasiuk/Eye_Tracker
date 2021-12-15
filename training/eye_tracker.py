from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import tensorflow as tf
import json
import sys
import time
import numpy as np
import cv2
from datetime import datetime

from preprocessing import preprocessor
from model import eye_tracker, eye_tracker_simple
from evaluate import evaluation
from xai.heatmap import HeatMap, HeatMapAlgorithm


def create_callbacks(checkpoint_dir, patience):
    early_stopper = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=patience,
                                verbose=1,
                                mode='auto')


    checkpoint_dir += datetime.now().strftime("%H_%M_%S")
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')
    os.mkdir(checkpoint_dir)
    checkpointer = ModelCheckpoint(checkpoint_dir + '/' + "eye_tracker" +
                                '-E{epoch:02d}-L{loss:.5f}-VL{val_loss:.5f}.hdf5',
                                monitor='val_loss', save_best_only=True, verbose=True)

    tfboard = TensorBoard(log_dir='./logs/')



    return [early_stopper, checkpointer, tfboard], checkpoint_dir

def get_filepath_of_last_weights_file(checkpoint_dir):
    unsorted_list = [os.path.join(checkpoint_dir,x) for x in os.listdir(checkpoint_dir)
                     if x.startswith("eye_tracker" + '-') and x.endswith(".hdf5")]
    weights_path = None if len(unsorted_list) == 0 else \
                   sorted(unsorted_list, key=lambda p: os.path.exists(p) and os.stat(p).st_mtime)[-1]
    return weights_path

def save_model(model, name):
    print('Saving the model: %s' % name)

    json_string = model.to_json()
    if not os.path.isdir('./release'):
        os.mkdir('./release')
    json_name = 'architecture_' + name + '.json'
    open(os.path.join('./release', json_name), 'w').write(json_string)

    weight_name = 'model_weights_' + name + '.h5'
    model.save_weights(os.path.join('release', weight_name), overwrite=True)

    tflite_ready_name = 'tflite_ready_' + name + '.h5'
    model.save(os.path.join('./release', tflite_ready_name))

def multiple_inputs_data_generator(X1, X2, y, batch_size):
    gen = ImageDataGenerator(rotation_range=2.5,width_shift_range=0.025,height_shift_range=0.025,
                                 brightness_range=None,horizontal_flip=False,zoom_range=0.025)
    genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=1)
    genX2 = gen.flow(X2, y, batch_size=batch_size, seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X1i[1]

def denormalize(imgs):
    return (imgs * 255).astype(np.uint8)

def save_images(images, name, convert_from_gray = False):
    i = 0
    for img in images:
        if convert_from_gray:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        cv2.imwrite("./XAI/" + name + "_" + str(i) + ".jpg", img)
        i = i + 1

def train(data, last_checkpoint_dir = None, xai_analysis = False):
    # odczytanie konfiguracji
    batch_size = data["batch_size"]
    checkpoint_dir = data["checkpoint_dir"]

    model = eye_tracker() # inicjalizacja modelu
    
    preproc = preprocessor(data) # inicjalizacja modułu preprocesora

    time_load_data_start = time.time() # ropoczęcie pomiaru czasu ładowania danych

    preproc.prepare_data() # ładowanie zbioru danych do pamięci oraz obróbka zdjęć

    # wybranie potrzebnych na czas treningu i/lub ewaluacji danych
    X_train_images = preproc.X_train_images
    X_test_images = preproc.X_test_images
    X_validation_images = preproc.X_validation_images

    X_train_array = preproc.X_train_array
    X_test_array = preproc.X_test_array
    X_validation_array = preproc.X_validation_array


    y_train = preproc.y_train
    y_test = preproc.y_test
    y_val = preproc.y_validation

    # transpozycja pierwszych dwóch osi z formatu (Ilość, Rodzaj_obrazu, Wymiary...)
    # na format (Rodzaj_obrazu, Ilość, Wymiary...)
    X_train_images = np.transpose(X_train_images,(1,0,2,3,4)) 
    X_validation_images = np.transpose(X_validation_images,(1,0,2,3,4))
    X_test_images = np.transpose(X_test_images,(1,0,2,3,4))

    time_load_data_end = time.time() # zakończenie pomiaru czasu ładowania zbioru danych

    callbacks, checkpoint_dir = create_callbacks(checkpoint_dir, data["patience"]) # przygotowanie funkcji sterujacych uczeniem

    time_training_start = time.time() # ropoczęcie pomiaru czasu treningu 

    if last_checkpoint_dir is None:
        history = model.fit([X_train_images[0], X_train_images[1], X_train_array], y_train, batch_size, 
                        steps_per_epoch=int((len(X_train_images[0])+batch_size-1)/batch_size),
                        validation_data=([X_validation_images[0], X_validation_images[1], X_validation_array], y_val),
                        epochs=data["epochs"],
                        verbose=1,
                        callbacks=callbacks)


    time_training_end = time.time() # zakończenie pomiaru czasu treningu

    # ładowanie danych z ostatniego punktu kontrolnego - posiada najlepszą wartość funkcji straty dla znioru walidacyjnego
    if last_checkpoint_dir is not None:
        print("loading previous checkpoint from ", data["checkpoint_dir"]+last_checkpoint_dir)
        model.load_weights(get_filepath_of_last_weights_file(data["checkpoint_dir"]+last_checkpoint_dir))
        save_model(model, "eye_tracker") # Zapisanie modelu w formie grafu (.h5)
    else:
        model.load_weights(get_filepath_of_last_weights_file(checkpoint_dir))
        save_model(model, "eye_tracker") # Zapisanie modelu w formie grafu (.h5)
    
    if xai_analysis:
        explainer = HeatMap(img_format='rgb') # stworzenie obiektu klasy odpowiedzialnej za tworzenie map ciepła

        # Utworzenie map ciepła dla oka
        heatmap_eye = explainer.create_heatmap(model,
                                    X_test_images,
                                    HeatMapAlgorithm.GradCAM,
                                    use_smoothing=False,
                                    selected_output_index=0,
                                    convolution_layer_name = "dropout_7", # Ta warstwa odpowiada gałąź oka
                                    verbose=True)

        # Utworzenie map ciepła dla nosa
        heatmap_nose = explainer.create_heatmap(model,
                                    X_test_images,
                                    HeatMapAlgorithm.GradCAM,
                                    use_smoothing=False,
                                    selected_output_index=0,
                                    convolution_layer_name = "dropout_15", # Ta warstwa odpowiada gałąź nosa
                                    verbose=True)

        # denormalizacja zdjęć zbioru testowego
        denormalized_imgs1 = denormalize(X_test_images[0])
        denormalized_imgs2 = denormalize(X_test_images[1])

        # utworzenie nałożonych map ciepła na oryginalne zdjęcia
        overlay_eye = explainer.create_overlay(heatmap_eye, denormalized_imgs1, alpha=0.6)
        overlay_nose = explainer.create_overlay(heatmap_nose, denormalized_imgs2, alpha=0.6)

        # utworzenie masek 
        masks_eye = explainer.create_mask(heatmap_eye, denormalized_imgs1, percentile=50)
        masks_nose = explainer.create_mask(heatmap_nose, denormalized_imgs2, percentile=50)

        # denormalizacja map ciepła, aby można je było zapisać
        heatmap_eye = denormalize(heatmap_eye)
        heatmap_nose = denormalize(heatmap_nose)

        # zapis zdjęć
        save_images(denormalized_imgs1, "original_eye")
        save_images(denormalized_imgs2, "original_nose")
        save_images(heatmap_eye, "heatmap_eye")
        save_images(heatmap_nose, "heatmap_nose")
        save_images(overlay_eye, "overlay_eye")
        save_images(overlay_nose, "overlay_nose")
        save_images(masks_eye, "masks_eye")
        save_images(masks_nose, "masks_nose")

    # obliczenie poszczególnych składowych czasu uczenia 
    training_time_seconds = time_training_end - time_training_start
    training_time_hours = training_time_seconds//3600
    training_time_minutes = (training_time_seconds - training_time_hours*3600)//60
    leftover_seconds = training_time_seconds-(60*training_time_minutes) - (3600 * training_time_hours)
    training_time = str(training_time_hours) + " hours, " + str(training_time_minutes) + " minutes and " + str(leftover_seconds) + " seconds"

    eval = evaluation(model, data) # przygotowanie obiektu ewaluacji
    eval.evaluate_model() # ewaluacja modelu

    # Przykładowe predykcje do naocznego określenia, czy proces uczenia przebiegł subiektywnie poprawnie
    for x in range(15):
        print("for label ", str(y_test[x]), " predicted ",\
             model.predict([X_test_images[0][x].reshape(-1,64,64,3),\
                            X_test_images[1][x].reshape(-1,64,64,3),\
                            X_test_array[x].reshape(-1,6)]))
    print("Preprocessing took ", time_load_data_end - time_load_data_start, " seconds, while training took in total ", training_time)

if __name__ == "__main__":
    last_checkpoint_dir = None
    xai_analysis = False
    if len(sys.argv) >= 3 :
        last_checkpoint_dir = sys.argv[2]
    if len(sys.argv) == 4 :
        xai_analysis = sys.argv[3]

    file = open(sys.argv[1])
    data = json.load(file)
    train(data, last_checkpoint_dir, xai_analysis)
    file.close()
    