import numpy as np
import cv2
import argparse
import pickle
from imutils.object_detection import non_max_suppression as nms
import matplotlib.pyplot as plt

# Classes detected by model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def detect_people(frame, winStride, scale, dnn):

    if dnn is False:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        (rects, weights) = hog.detectMultiScale(frame, winStride=winStride, padding=(4, 4), scale=scale, hitThreshold=0.4)
        rects = nms(rects, probs=None, overlapThresh=0.65)

    else:
        rects = []
        prototxt = "../model/MobileNetSSD_deploy.prototxt"
        caffeModel = "../model/MobileNetSSD_deploy.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt, caffeModel)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(rgb, 0.007843, (w, h), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:

                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # rescale detection box size to frame size
                (startX, startY, endX, endY) = box.astype("int")

                x = startX
                y = startY
                w = endX - startX
                h = endY - startY

                rects.append((x, y, w, h))

    return rects


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    xB = min(boxA[2], boxB[2])
    yA = max(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])

    # calculate the area of intersection of the two bounding boxes
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # calculate the area of both the ground truth and prediction
    # bounding-box
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-ap", "--annotationsFile", required=False, default="annotations/frame_annotations.pickle")
    ap.add_argument("-v", "--video", required=False, default="../testVideo/IMG_0330.MOV")
    args = vars(ap.parse_args())

    with open(args["annotationsFile"], 'rb') as annotations_list_file:
        annotations_list = pickle.load(annotations_list_file)

    n = 0
    IoUTotalAvg = 0
    totalFalsePositives = 0
    totalFalseNegatives = 0
    totalTruePositives = 0
    totalAccuracy = 0

    precision = []
    recall = []
    anno = 0

    while True:
        n += 1

        if anno == len(annotations_list) - 1:
            break

        IoU = []
        iouTotal = 0

        im = annotations_list[anno][0]

        gtbbox = annotations_list[anno][1]
        predbbox = detect_people(im, (4, 4), 1.25, dnn=True)

        print("predicted bboxes: {} ".format(predbbox))
        print("ground truth bboxes: {} ".format(gtbbox))

        for gtbox in gtbbox:

            gtX1 = gtbox[0]
            gtY1 = gtbox[1]
            gtX2 = gtbox[0] + gtbox[2]
            gtY2 = gtbox[1] + gtbox[3]

            cv2.rectangle(im, (gtX1, gtY1),
                          (gtX2, gtY2), (0, 255, 0), 2)

        for predbox in predbbox:

            predX1 = predbox[0]
            predY1 = predbox[1]
            predX2 = predbox[2] + predbox[0]
            predY2 = predbox[3] + predbox[1]

            cv2.rectangle(im, (predX1, predY1),
                          (predX2, predY2), (0, 0, 255), 2)

        for i in gtbbox:
            iouMax = 0

            for j in predbbox:

                iou = bb_intersection_over_union((i[0], i[1], i[2] + i[0], i[3] + i[1]),
                                                 (j[0], j[1], j[2] + j[0], j[3] + j[1]))

                if iou > iouMax:
                    iouMax = iou

            if iouMax > 0.3:
                IoU.append(iouMax)

        if len(predbbox) > len(gtbbox):
            falsePositives = len(predbbox) - len(IoU)

        else:
            falsePositives = 0

        for k in IoU:
            iouTotal = iouTotal + k

        if len(IoU) > 0:
            iouAvg = iouTotal/len(IoU)

        else:
            iouAvg = 0

        # cv2.putText(im, "IoU: {:.4f}".format(iou), (10, 30),
        #          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        truePositives = len(IoU)
        falseNegatives = len(gtbbox) - len(IoU)

        print("Number of false Positives: {}".format(falsePositives))
        print("Number of false Negatives: {}".format(falseNegatives))
        print("Number of true Positives: {}".format(truePositives))
        print("IoU's : {}".format(IoU))
        print("Avg IoU: {}".format(iouAvg))

        IoUTotalAvg += iouAvg
        totalFalseNegatives += falseNegatives
        totalFalsePositives += falsePositives
        totalTruePositives += truePositives
        totalAccuracy += truePositives/len(gtbbox)

        if truePositives != 0 or falsePositives != 0:
            precision.append(truePositives/(truePositives + falsePositives))
        else:
            precision.append(0)

        if truePositives != 0 or falseNegatives != 0:
            recall.append(truePositives/(truePositives + falseNegatives))
        else:
            recall.append(0)

        cv2.imshow("Image", im)
        cv2.waitKey(0)
        anno += 1

    IoUAverage = IoUTotalAvg/n
    avgFalseNegatives = totalFalseNegatives/n
    avgFalsePositives = totalFalsePositives/n
    avgTruePositives = totalTruePositives/n
    avgAccuracy = totalAccuracy/n

    plt.plot(recall, precision, 'bo')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

    print("\nIoU average: {}".format(IoUAverage))
    print("average false negatives: {}".format(avgFalseNegatives))
    print("average false positives: {}".format(avgFalsePositives))
    print("average true positives: {}".format(avgTruePositives))
    print("Accuracy: {}".format(avgAccuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))

