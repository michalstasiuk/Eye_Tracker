import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

target_height = 3000

mouth_points = [0,11,12,13,15,16,17,18,37,38,39,40,41,43,61,72,73,74,76,77,80,81,82,83,84,85,
                86,87,88,89,90,91,95,96,165,178,179,180,181,182,183,184,185,
                200,201,202,211,267,268,269,270,271,272,273,291,302,303,304,306,
                307,310,311,312,313,314,315,316,317,318,319,320,321,324,325,391,402,
                403,404,405,406,407,409,421]

const_eye_points = [341,445]
const_nose_points = [57,355]

class face_mesh():
    def __init__(self) :
        self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesher = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def calculate_distance(self, face, image):
        distance_vector = []
        #for landmark in face:
            

    def draw_landmarks(self, image, face):
        i = 0
        for landmark in face:
            #print("drawing point number ", i) landmark[0][0], landmark[0][1]
            cv2.putText(image,str(i), (landmark[0], landmark[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            image = cv2.circle(image, (landmark[0], landmark[1]), 2, (255,0,0), 1)
            #cropped = image[300:1300, 100:1000]
            i = i + 1
        cv2.imwrite("test.jpg", image)
        #cv2.waitKey(0)


    #def calculate_distance

    def mesh_if_exist(self, image, name):
        width = int(image.shape[1])
        height = int(image.shape[0])
        proportion = height/width
        target_width = int(target_height/proportion)
        image = cv2.resize(image, (target_width, target_height))
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.face_mesher.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            print("***************************")
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)
                cv2.imwrite(name, image)
                if cv2.waitKey(0) & 0xFF == 27:
                    break
            return True
        else:
            return False

    def get_crop_dims(self, landmarks):
        x = []
        y = []
        for landmark in landmarks:
            x.append(landmark[0])
            y.append(landmark[1])
        min_x = min(x)
        min_y = min(y)
        max_x = max(x)
        max_y = max(y)
        return [min_x, min_y, max_x, max_y]
        
        

    def get_landmarks(self, image):
        width = int(image.shape[1])
        height = int(image.shape[0])
        proportion = height/width
        target_width = int(target_height/proportion)
        #image = cv2.resize(image, (target_width, target_height))
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.face_mesher.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        face = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:  
                for id,lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = image.shape
                    x,y,z = int(lm.x*iw), int(lm.y*ih), int(lm.z*255)
                    face.append([x,y])

        eye_landmarks = []
        nose_landmarks = []
        #self.draw_landmarks(image, face)
        for corner in const_eye_points:
            if len(face) < corner:
                print ("empty image!!!!!!!!!")
                return [[0,0,0,0],[0,0,0,0]]
            eye_landmarks.append(face[corner])
        for corner in const_nose_points:
            if len(face) < corner:
                print ("empty image!!!!!!!!!")
                return [[0,0,0,0],[0,0,0,0]]
            nose_landmarks.append(face[corner])

        return self.get_crop_dims(eye_landmarks), self.get_crop_dims(nose_landmarks)

    def cropp_by_eye_landmarks(self,image):
        eye_crop_dims, nose_crop_dims = self.get_landmarks(image)
        cropped_eye = image[eye_crop_dims[1] : eye_crop_dims [3], eye_crop_dims[0] : eye_crop_dims [2]]
        cropped_nose = image[nose_crop_dims[1] : nose_crop_dims [3], nose_crop_dims[0] : nose_crop_dims [2]]

        ih, iw, ic = image.shape
        high_level_info = []
        high_level_info.append(float(eye_crop_dims[0]-nose_crop_dims[0])/float(iw))
        high_level_info.append(float(eye_crop_dims[3]-nose_crop_dims[3])/float(ih))
        high_level_info.append(float(eye_crop_dims[0])/float(iw))
        high_level_info.append(float(eye_crop_dims[3])/float(ih))
        high_level_info.append(float(nose_crop_dims[0])/float(iw))
        high_level_info.append(float(nose_crop_dims[3])/float(ih))
        
        return [cropped_eye, cropped_nose], high_level_info