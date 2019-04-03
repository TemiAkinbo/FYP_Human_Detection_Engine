import cv2
import numpy as np
import pickle
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--imagesPath", required=False, default="../../Datasets/persons")
ap.add_argument("-op", "--outputFile", required=False, default="annotations.pickle")

args = vars(ap.parse_args())

# load list with images to be annotated
images = glob.glob(args["imagesPath"] + "/*.bmp")

# print(images)

# define 'Ánnotation' object
# Annotation = namedtuple("Annotations", ["image_path", "gt"])

examples = []

for image in images[: 50]:

    # Read Image
    im = cv2.imread(image)

    print

    bboxes = []
    while True:
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        fromCenter = False
        bbox = cv2.selectROI(image, im, fromCenter=fromCenter)
        bboxes.append(bbox)
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if k == 113:  # q is pressed
            break

    print(bboxes)

    #create an annotation object
    annotation = (image, bboxes)

    # add annotation to list of images with annotations
    examples.append(annotation)

    for bbox in bboxes:
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

    cv2.imshow("Image", im)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

annotation_list_file = open('annotations.pickle', 'wb')
pickle.dump(examples, annotation_list_file)
annotation_list_file.close()
