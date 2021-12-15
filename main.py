import cv2
import sys
import json 

from face_control import Face_Control

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("provide configuration.json file, exiting")
        exit()
    file = open(sys.argv[1])
    config = json.load(file)

    if len(sys.argv) == 4:
        video_analysis = sys.argv[2]
        video_visualize = sys.argv[3]
        face_control = Face_Control(config, video_analysis, video_visualize)
        print("***********************************************")
        face_control.run()
    else :
        face_control = Face_Control(config)
        face_control.run()
    



