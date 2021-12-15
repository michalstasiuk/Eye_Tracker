from face_mesh import face_mesh
import cv2
import sys

path = sys.argv[1]
#name = sys.argv[2]

image = cv2.imread(path)

mesher = face_mesh()
mesher.get_landmarks(image)
# images = mesher.cropp_by_eye_landmarks(image)

# preprocessed = [cv2.resize(images[0], (64,64)),\
#                 cv2.resize(images[1], (64,64))]

# cv2.imwrite("eye.jpg", preprocessed[0])
# cv2.imwrite("nose.jpg", preprocessed[1])

# cap = cv2.VideoCapture(0)
#mesher.cropp_by_eye_landmarks(image)
# while cap.isOpened():
#     success, image = cap.read()
#     cv2.imshow("eye", mesher.cropp_by_eye_landmarks(image))
#     key = cv2.waitKey(0) & 0xFF
#     if key == 27:
#         print("pressed escape, exiting")
#         break

