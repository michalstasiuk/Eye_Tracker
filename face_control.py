import cv2
from eye_tracker import Eye_Tracker
from threading import Thread
import numpy as np
from statistics import mode
from pykeyboard import PyKeyboard
from face_detection import face_detection

def eye_tracker_worker(eye_tracker):
    pass

class Low_Pass_Filter:
    def __init__(self, config):
        self.sample_period = config["sample_period"]
        self.cutoff_freq = config["cutoff_freq"]
        # inicjalizacja wartości parametru alfa
        self.alpha = 1-np.exp(-self.sample_period * self.cutoff_freq)
        self.old_data = 0.0

    def filter_new_value(self, value):
        # filtracja nowej wartości zgodnie z opisanym wzorem w pracy dyplomowej
        self.old_data = value * self.alpha + (1-self.alpha) * self.old_data
        return self.old_data
        

class Face_Control:
    def __init__(self, config, video_analysis= "0", video_visualize = "0"):
        # inicjalizacja obiektu wykrwyania ruchu gałek ocznych
        self.eye_tracker = Eye_Tracker(config)
        self.face_detection = face_detection()
        
        #otwarcie strumienia wideo, nagrania lub dostępnej kamery
        if video_analysis == "0":
            self.embedded_camera = True
            self.cap = cv2.VideoCapture(int(video_analysis))
        else:
            self.embedded_camera = False
            self.cap = cv2.VideoCapture(video_analysis)
            self.visualize = cv2.VideoCapture(video_visualize)
            #self.out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MPEG'), 20.0, (2498,1080))
            self.count = 0

        self.filter = Low_Pass_Filter(config["filter"])  

        self.last_angles_count = config["REM_detection"]["last_angles_count"]
        self.rapid_eye_movement_cumulative_difference_threshlold = config["REM_detection"]["threshold"]
        self.angles = [0.5]*self.last_angles_count #inicjalizacja tablicy zmierzonych kątów n takimi samymi wartościami
        self.filtered_angles = [0.5]*self.last_angles_count #inicjalizacja tablicy zmierzonych kątów n takimi samymi wartościami
        self.last_filtered_angle = 0.5

        self.y_pred = []
        self.y_pred_filtered = []

        self.keyboard = PyKeyboard() # inicjalizacja obiektu emulatora klawiatury

    def run(self):
        success, image = self.cap.read()

        while success : # jeżeli dla każdego kolejnego odczytu ze strumienia jest sukces
           
            #image = cv2.flip(image, 1)
            angle = self.eye_tracker.run(image) # odczytaj aktualny kąt wychylenia gałki ocznej
            if not self.embedded_camera:
                v_success, visualize_image = self.visualize.read()
                self.last_image_analysis = image
                if v_success:
                    self.last_image = visualize_image
            else:
                self.last_image = image
            filtered_angle = self.filter.filter_new_value(angle) # przefiltruj kąt
            # określ, czy kąt w kątekście poprzednich ruchów oraz ruchów przefiltrowanych nie wskazuje na nieintencjonalny ruch gałki ocznej
            is_rapid_eye_movement = self.is_rapid_eye_movement(angle, filtered_angle) 

            if is_rapid_eye_movement is not True: # jeżeli kąt NIE wskazuje na nieintencjonalny ruch gałki ocznej
                # Teraz można przypisać przefiltrowaną wartość, funkcja is_rapid_eye_movement operuje na poprzednich wartościach
                self.last_filtered_angle = filtered_angle 
                self.actuate_turning(filtered_angle) #skręć pojazdem
            else:
                self.actuate_turning(filtered_angle, True) # utrzymaj poprzedni kurs

            self.y_pred.append(angle)
            self.y_pred_filtered.append(self.last_filtered_angle)
            self.visualize_actuation(angle, is_rapid_eye_movement)
            #self.visualize_side_by_side(angle, is_rapid_eye_movement)
            # cv2.imshow("face_control demo", self.last_image)
            # key = cv2.waitKey(1) & 0xFF
            # if key == 27:
            #     print("pressed escape, exiting")
            #     break
            success, image = self.cap.read()
        print("Unable to continue taking frames from provided source, exiting")
        self.make_clusterization(self.y_pred, self.y_pred_filtered)
        #self.out.release()
        exit()
    
    def is_rapid_eye_movement(self, angle, filtered_angle):
        # aktualizacja list z ostatnimi filtrowanymi i niefiltrowanymi wartoścami wychylenia gałki ocznej
        self.angles.append(angle)
        self.angles = self.angles[-self.last_angles_count:]
        self.filtered_angles.append(filtered_angle)
        self.filtered_angles = self.filtered_angles[-self.last_angles_count:]

        cumulative_difference = 0
        for i in range(len(self.angles)):
            # dla każdej iteracji oblicz odległość pomiędzy kątem filtrowanym i niefiltrowanym, dodaj do sumy.
            cumulative_difference += np.abs(self.angles[i] - self.filtered_angles[i])

        print("cumulative_difference: ", cumulative_difference)

        # jeżeli suma jest większa niż wartość graniczna
        if cumulative_difference >= self.rapid_eye_movement_cumulative_difference_threshlold:
            return True
        else:
            return False

    def actuate_turning(self, angle, release = False):
        # release powinine wskazywać na utrzymanie poprzedniego kursu
        pass
        # if release:
        #     self.keyboard.release_key(self.keyboard.left_key)
        #     self.keyboard.release_key(self.keyboard.right_key)

        # if angle < 0.3:
        #     self.keyboard.press_key(self.keyboard.left_key)
        # else:
        #     if angle > 0.3:
        #         self.keyboard.press_key(self.keyboard.right_key)
        #     else:
        #         self.keyboard.release_key(self.keyboard.left_key)
        #         self.keyboard.release_key(self.keyboard.right_key)
       

    def visualize_actuation(self, angle, is_rapid_eye_movement):
        margin = 15 #pixels
        img = self.last_image
        #img = np.zeros([1080,1920,3],dtype=np.uint8)
        #img.fill(255)
        cv2.putText(img, str(self.last_filtered_angle), (margin, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(img, "is_rapid_eye_movement: " + str(is_rapid_eye_movement), (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        
        height, width, channels = img.shape
        
        x_value_filtered = int(margin + (width-margin) * self.last_filtered_angle)
        x_value_raw = int(margin + (width-margin) * angle)

        cv2.line(img, (margin, height-100), (width-margin, height-100), (255,255,255), 2) # Skala w kolorze białym
        cv2.line(img, (x_value_filtered, height-90), (x_value_filtered, height-110), (255,0,0), 2) # Linia skupienia - filtrowana
        cv2.line(img, (x_value_raw, height-90), (x_value_raw, height-110), (255,255,255), 2) # Linia skupienia - niefiltrowana
        cv2.imshow("face_control_demo.jpg", img)
        if not self.embedded_camera:
                cv2.imshow("from analyzing video", self.last_image_analysis)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            print("pressed escape, exiting")
            exit()

    def visualize_side_by_side(self, angle, is_rapid_eye_movement):
        margin = 15 #pixels
        img = self.last_image
        self.count += 1
        cv2.putText(img, str(self.last_filtered_angle), (margin, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
        cv2.putText(img, "is_rapid_eye_movement: " + str(is_rapid_eye_movement), (margin, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
        
        height, width, channels = img.shape
        
        x_value_filtered = int(margin + (width-margin) * self.last_filtered_angle)
        x_value_raw = int(margin + (width-margin) * angle)

        cv2.line(img, (margin, height-100), (width-margin, height-100), (255,255,255), 4) # Skala w kolorze białym
        cv2.line(img, (x_value_filtered, height-70), (x_value_filtered, height-130), (255,0,0), 4) # Linia skupienia - filtrowana
        cv2.line(img, (x_value_raw, height-70), (x_value_raw, height-130), (255,255,255), 4) # Linia skupienia - niefiltrowana
        
        face = self.face_detection.crop_face(self.last_image_analysis)
        height, width,_ = img.shape 
        target_width = int(height/1.73)
        face = cv2.resize(face, (target_width, height))
        
        composed = np.concatenate((img, face), axis=1)
        resized = cv2.resize(composed, (int(composed.shape[1]/2), int(composed.shape[0]/2)))
        cv2.imshow("from analyzing video", resized)
        #self.out.write(composed)
        cv2.imwrite("zdjecia_obrona/" + str(self.count) + ".jpg", composed)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("pressed escape, exiting")
            exit()

    def make_clusterization(self, angles, filtered_angles):
        img = self.last_image_analysis
        img.fill(255)
        margin = 20
        margin_high = margin
        height, widht, _ = img.shape
        cv2.line(img, (margin, height-margin_high), (widht-margin, height-margin_high), (255,255,255), 1)

        next_lane = (widht-2*margin)//10
        first_line_start = margin + next_lane
        
        cv2.line(img, (first_line_start+0*next_lane, height - margin_high), (first_line_start+0*next_lane, margin_high), (255,0,0), 1,)
        cv2.line(img, (first_line_start+1*next_lane, height - margin_high), (first_line_start+1*next_lane, margin_high), (255,255,0), 1,)
        cv2.line(img, (first_line_start+2*next_lane, height - margin_high), (first_line_start+2*next_lane, margin_high), (255,0,255), 1,)
        cv2.line(img, (first_line_start+3*next_lane, height - margin_high), (first_line_start+3*next_lane, margin_high), (0,255,0), 1,)
        cv2.line(img, (first_line_start+4*next_lane, height - margin_high), (first_line_start+4*next_lane, margin_high), (0,0,255), 1,)
        cv2.line(img, (first_line_start+5*next_lane, height - margin_high), (first_line_start+5*next_lane, margin_high), (0,255,255), 1,)
        cv2.line(img, (first_line_start+6*next_lane, height - margin_high), (first_line_start+6*next_lane, margin_high), (128,192,64), 1,)
        cv2.line(img, (first_line_start+7*next_lane, height - margin_high), (first_line_start+7*next_lane, margin_high), (64,128,192), 1,)
        cv2.line(img, (first_line_start+8*next_lane, height - margin_high), (first_line_start+8*next_lane, margin_high), (192,64,128), 1,)

        for i in range(len(angles)):
            cv2.circle(img, (int(margin+(widht-2*margin)*angles[i]), int(height - margin_high - (height-2*margin_high)/len(angles)*i)), 1, (255,0,0), 2)
        for i in range(len(filtered_angles)):
            cv2.circle(img, (int(margin+(widht-2*margin)*filtered_angles[i]), int(height - margin_high- (height-2*margin_high)/len(filtered_angles)*i)), 1, (0,255,0), 2)
        cv2.imwrite("clusterization.jpg", img)
        

    
    
        
        
