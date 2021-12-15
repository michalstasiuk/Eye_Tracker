import cv2
import sys
from pathlib import Path
import glob

def save_each_n_image_as_unique(folder, period):
    files = glob.glob(folder + '/*.jpg') # znalezienie każdego pliku w folderze
    images = []
    for image_path in files: # odczytanie każdego zdjęcia z scieżki
        print("loading file", image_path)
        images.append(cv2.imread(image_path))

    unique_images = images[::period] # wycięcie co n-tego zdjęcia z tablicy

    print(len(unique_images), " unique images out of ", len(files))

    i = 0
    for image in unique_images:
        name = folder  + "/" + str(i) + "_unique" + ".jpg" # utworzenie unikalnej nazwy zdjęcia
        print("saving file ", name)
        cv2.imwrite(name, image) # zapisanie zdjęcia
        i = i +1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("provide folder with images and which set type, exiting")
        exit()

    folder = sys.argv[1] # odczytanie folderu głownego z którego należy wytypować tylko co n plik
    period = 10 # co które zdjęcie ma być zapisywane
    if len(sys.argv) == 3:
        period = int(sys.argv[2])

    # zapisywanie zdjęcia co n (period) z wygenerowanej ścieżki
    save_each_n_image_as_unique(folder + "0.1", period)
    save_each_n_image_as_unique(folder + "0.2", period)
    save_each_n_image_as_unique(folder + "0.3", period)
    save_each_n_image_as_unique(folder + "0.4", period)
    save_each_n_image_as_unique(folder + "0.5", period)
    save_each_n_image_as_unique(folder + "0.6", period)
    save_each_n_image_as_unique(folder + "0.7", period)
    save_each_n_image_as_unique(folder + "0.8", period)
    save_each_n_image_as_unique(folder + "0.9", period)


