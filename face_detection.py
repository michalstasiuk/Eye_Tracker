import cv2
import mediapipe as mp

scale_percent = 50

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

class face_detection():
    def __init__(self) :
        self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def crop_face(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.face_detector.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detection = str(results.detections[0])
        
        xmin_tmp = detection.split("xmin: ", 1)[1]
        ymin_tmp = detection.split("ymin: ", 1)[1]
        width_tmp = detection.split("width: ", 1)[1]
        height_tmp = detection.split("height: ", 1)[1]

        xmin_tmp = xmin_tmp.split("ymin: ", 1)[0]
        ymin_tmp = ymin_tmp.split("width: ", 1)[0]
        width_tmp = width_tmp.split("height: ", 1)[0]
        height_tmp = height_tmp.split("}", 1)[0]

        img_height, img_widht, _ = image.shape

        xmin = float(xmin_tmp)
        ymin = float(ymin_tmp)
        width = float(width_tmp)
        height = float(height_tmp)

        xmin = int(xmin*img_widht)
        ymin = int(ymin*img_height)
        width = int(width*img_widht)
        height = int(height*img_height)

        print("xmin", xmin)
        print("ymin", ymin)
        print("width", width)
        print("height", height)
        cropped = image[ymin:ymin+height, xmin:xmin+width]
        
        return cropped