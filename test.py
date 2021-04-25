# import the necessary libs
import numpy as np
import argparse
import time
import cv2
import os
import sys
from PIL import Image

cwd = os.getcwd()
print("cwd", cwd)

sys.path.append(cwd + "/" + "libs")
sys.path.append(cwd + "/" + "libs/insight_face/InsightFace_Pytorch")
sys.path.append(cwd + "/" + "libs/insight_face")
print("path sys: ")
for line in sys.path:
    print(line)

from libs import yolo_detection
from libs.insight_face.InsightFace_Pytorch import face_verify

#yolo_detection.test_import()

net, ln = yolo_detection.init_net()
print("TEST - yolo detection NET initted")

conf = face_verify.init_config()
mtcnn = face_verify.init_MTCNN()
learner = face_verify.init_learner(conf)
targets, names = face_verify.init_facebank(conf=conf, learner=learner, mtcnn=mtcnn)

def test_face_detection():
    image = cv2.imread("test/input/anhhiep_testmulti.jpg")
    boxes, confidences, classIDs, detected_image = yolo_detection.detect_bboxes(net, ln, image)
    cv2.imwrite("test/output/face_detection.png", detected_image)

    for i in range(len(boxes)):
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        crop_face = image[y:y+h,x:x+w].copy()
        cv2.imwrite("test/output/output_crop/crop_face_{}.png".format(str(i)), crop_face)

def test_face_recognition_test_loaded_images():
    print("START TEST FACE RECOG")
    face_verify.verify_images_test(conf=conf, learner=learner, targets=targets)

def test_face_detection_recognition():
    print("START TEST FACE DETECTION RECOG")
    
    #image = cv2.imread("test/input/anhhiep_testmulti.jpg")
    #image = cv2.imread("test/input/DJI_0039_Moment_2_crop.jpg")
    #image = cv2.imread("test/input/DJI_0039_Moment_2.jpg")
    #image = cv2.imread("test/input/DJI_0098_Moment_Moment.jpg")
    image = cv2.imread("test/input/DJI_0098_1.jpg")

    boxes, confidences, classIDs, detected_image = yolo_detection.detect_bboxes(net, ln, image)
    cv2.imwrite("test/output/face_detection.png", detected_image)
    
    START_TIME = time.time()
    faces = []
    for i in range(len(boxes)):
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # crop and save face
        DELTA_y = int(0.1 * h)
        DELTA_x = int(0.2 * w)
		#crop_face = image[y:y+h,x:x+w].copy()  
        crop_face = image[y-DELTA_y*2:y+h+DELTA_y,x-DELTA_x:x+w+DELTA_x].copy()

        #cv2.imwrite("test/output/output_crop/crop_face_{}.png".format(str(i)), crop_face)

        pillow_image = Image.fromarray(cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB))
        face = pillow_image.resize((112,112))
        faces.append(face)
    face_ids = range(len(faces))
    face_verify.verify_faces(conf=conf, learner=learner, targets=targets, faces=faces, face_ids=face_ids)
    
    END_TIME = time.time()
    print("time cost: ", END_TIME - START_TIME)
#test_face_detection()
#face_verify.test_import()
#test_face_recognition_test_loaded_images()
test_face_detection_recognition()