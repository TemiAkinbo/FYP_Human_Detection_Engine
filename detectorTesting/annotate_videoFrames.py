import cv2
import numpy as np
import pickle
import argparse
import imutils


ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--videoPath", required=False, default="../testVideo/IMG_0330.MOV")
ap.add_argument("-op", "--outputFile", required=False, default="videoframe_annotations.pickle")
ap.add_argument("-s", "--scaleFactor", required=False, default=1, type=float)

args = vars(ap.parse_args())

video = cv2.VideoCapture(args["videoPath"])
scaleFactor = args["scaleFactor"]

grabbed, frame = video.read()

#image resize dimensions
resize_dim = int(frame.shape[0]/scaleFactor)

examples = []

for i in range(0, 120):

    grabbed, frame = video.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=resize_dim)

    bboxes = []
    while True:
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        fromCenter = False
        bbox = cv2.selectROI("frame", frame, fromCenter=fromCenter)
        bboxes.append(bbox)
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if k == 113:  # q is pressed
            break

    print(bboxes)

    #create an annotation object
    annotation = (bboxes)

    # add annotation to list of images with annotations
    examples.append(annotation)

    for bbox in bboxes:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

annotation_list_file = open(args["outputFile"], 'wb')
pickle.dump(examples, annotation_list_file)
annotation_list_file.close()
