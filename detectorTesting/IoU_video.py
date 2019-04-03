import numpy as np
import cv2
import argparse
import pickle
from imutils.object_detection import non_max_suppression as nms
import matplotlib.pyplot as plt
import csv
import datetime
import imutils
from imutils.video import FPS

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def detect_people(frame, winStride, scale , dnn):

    if dnn == False:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        (rects, weights) = hog.detectMultiScale(frame, winStride=winStride, padding=(4, 4), scale=scale, hitThreshold=0.4)
        rects = nms(rects, probs=None, overlapThresh=0.65)

    else:
        rects = []
        prototxt = "../mobilenet_ssd/MobileNetSSD_deploy.prototxt"
        caffeModel = "../mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt, caffeModel)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(rgb, 0.007843, (w, h), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.2:

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
    ap.add_argument("-ap", "--annotationsFile", required=False, default="annotations/videoframe_annotations.pickle")
    ap.add_argument("-v", "--video", required=False, default="../testVideo/IMG_0330.MOV")
    ap.add_argument("-s", "--scaleFactor", type=float, required=False,
                    default=1)
    args = vars(ap.parse_args())

    with open(args["annotationsFile"], 'rb') as annotations_list_file:
        annotations_list = pickle.load(annotations_list_file)

    # for calculating averages
    n = 0
    IoUTotalAvg = 0
    totalFalsePositives = 0
    totalFalseNegatives = 0
    totalTruePositives = 0
    totalAccuracy = 0

    precision = []
    recall = []
    anno = 0

    report = []

    resizeFactor = args["scaleFactor"]

    vs = cv2.VideoCapture(args["video"])
    videoFPS = vs.get(cv2.CAP_PROP_FPS)
    print("Video stream FPS: {:.2f}".format(videoFPS))

    grabbed, frame = vs.read()
    print(frame.shape)

    resizeDIMS = int(frame.shape[0]/resizeFactor), int(frame.shape[1]/resizeFactor)

    print(resizeDIMS)

    fps = FPS().start()

    while True:

        (grabbed, frame) = vs.read()

        if frame is None:
            break

        if anno < len(annotations_list):
            bbox = annotations_list[n]

        else:
            break

        n += 1

        IoU = []
        iouTotal = 0

        # im = cv2.imread(annotations_list[anno][0])

        frame = imutils.resize(frame, width=resizeDIMS[0], height=resizeDIMS[1])

        gtbbox = bbox
        predbbox = detect_people(frame, (2, 2), 1.15, dnn=True)

        fps.update()

        print("predicted bboxes: {} ".format(predbbox))
        print("ground truth bboxes: {} ".format(gtbbox))

        for gtbox in gtbbox:

            if not all(gtbox):
                continue

            iouMax = 0

            gtX1 = int(gtbox[0]/resizeFactor)
            gtY1 = int(gtbox[1]/resizeFactor)
            gtX2 = int((gtbox[0] + gtbox[2])/resizeFactor)
            gtY2 = int((gtbox[1] + gtbox[3])/resizeFactor)

            cv2.rectangle(frame, (gtX1, gtY1), (gtX2, gtY2), (0, 255, 0), 2)

            for predbox in predbbox:

                predX1 = predbox[0]
                predY1 = predbox[1]
                predX2 = predbox[2] + predbox[0]
                predY2 = predbox[3] + predbox[1]

                iou = bb_intersection_over_union((gtX1, gtY1, gtX2, gtY2),
                                                 (predX1, predY1, predX2, predY2))

                if iou > iouMax:
                    iouMax = iou

                cv2.rectangle(frame, (predX1, predY1),
                              (predX2, predY2), (0, 0, 255), 2)
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
        totalAccuracy += truePositives/(truePositives + falseNegatives)

        if truePositives != 0 or falsePositives != 0:
            prec = truePositives/(truePositives + falsePositives)
            precision.append(prec)
        else:
            precision.append(0)

        if truePositives != 0 or falseNegatives != 0:
            recall.append(truePositives/(truePositives + falseNegatives))
        else:
            recall.append(0)

        reportingDict = {
            "Frame": anno,
            "True Positives": truePositives,
            "False Negatives": falseNegatives,
            "False Positives": falsePositives,
            "Avg IoU": iouAvg,
            "Precision": prec
        }

        report.append(reportingDict)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        anno += 1

    fps.stop()

    IoUAverage = IoUTotalAvg/n
    avgFalseNegatives = totalFalseNegatives/n
    avgFalsePositives = totalFalsePositives/n
    avgTruePositives = totalTruePositives/n
    avgAccuracy = totalAccuracy/n

    # plt.plot(recall, precision, 'bo')
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.show()
    #
    print("\nIoU average: {}".format(IoUAverage))
    print("average false negatives: {}".format(avgFalseNegatives))
    print("average false positives: {}".format(avgFalsePositives))
    print("average true positives: {}".format(avgTruePositives))
    print("Accuracy: {}".format(avgAccuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("FPS: {:.2f}".format(fps.fps()))

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    filename = "EvalReports/eval_system_report_{}_.csv".format(now)

    with open(filename, mode='w') as csv_file:
        fieldnames = ['Frame', 'True Positives', 'False Negatives', 'False Positives', 'Avg IoU', 'Precision']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer = csv.writer(csv_file, delimiter=',')

        writer.writeheader()
        for line in report:
            writer.writerow(line)


