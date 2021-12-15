import cv2
import sys
from pathlib import Path

class key_parser():
    def __init__(self, project):
        self.path = project

    def parse_key_and_save_image(self, image, key, number):
        label = float(key - 48)/10 # generacja etykiety na podstawie kalwisza
        path = self.path + "/" + str(label) + "/" + str(number) + ".jpg" # generacja ścieżki zapisu
        print("Saving to file ", path)
        ret = cv2.imwrite(path, image) # zapisanie zdjęcia

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("provide name of output and video to annotate, exiting")
        exit()
        
    scene = sys.argv[1] # Nazwa scenerii (pacjent plus oświetlenie i jakość)

    # przygotowanie ścieżek zapisu zdjęć
    project = str("/home/stasiukm/EyeTracker/FaceControl/dataset/eye_tracker/" + scene)
    Path(project + "/0.1").mkdir(parents=True, exist_ok=True)
    Path(project + "/0.2").mkdir(parents=True, exist_ok=True)
    Path(project + "/0.3").mkdir(parents=True, exist_ok=True)
    Path(project + "/0.4").mkdir(parents=True, exist_ok=True)
    Path(project + "/0.5").mkdir(parents=True, exist_ok=True)
    Path(project + "/0.6").mkdir(parents=True, exist_ok=True)
    Path(project + "/0.7").mkdir(parents=True, exist_ok=True)
    Path(project + "/0.8").mkdir(parents=True, exist_ok=True)
    Path(project + "/0.9").mkdir(parents=True, exist_ok=True)

    # wybór źródła strumienia wideo
    if sys.argv[2] == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(sys.argv[2])

    parser = key_parser(project)

    frame_no = 0 
    while cap.isOpened(): # dopóki strumień otwarty
        success, image = cap.read()
        if not success:
            print("Video end or something went wrong, exiting")
            exit()
        cv2.imshow("image", cv2.flip(image,1)) # obróć horyzontalnie dla ułatwienia podglądu
        key = cv2.waitKey(0) & 0xFF # poczekaj na wciśnięcie klawisza (cyfry)
        if key == 27: # jeżeli escape, zakończ badanie
            print("pressed escape, exiting")
            break
        if key == 32: # jeżeli spacja, pomiń zapis (przydatne gdy jest przerwa pomiędzy seriami)
            print("pressed space, continuing without annotation")
            continue
        parser.parse_key_and_save_image(image, key, frame_no) # parsowanie klawisza oraz zapisanie zdjęcia z numer klatki frame_no
        print("proceeding to next frame")
        frame_no = frame_no + 1


