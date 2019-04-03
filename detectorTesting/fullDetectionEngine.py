from imutils.video import FPS
import numpy as np
import imutils
import dlib
import cv2
import argparse
import pickle
from imutils.object_detection import non_max_suppression as nms
import matplotlib.pyplot as plt


def dnnDetection(frame):

    rects = []
    prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
    caffeModel = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
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

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # rescale detection box size to frame size
            (startX, startY, endX, endY) = box.astype("int")

            x = startX
            y = startY
            w = endX - startX
            h = endY - startY

            rects.append((x, y, w, h))

    return rects


def hogDetection(frame, winStride, scale):

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (rects, weights) = hog.detectMultiScale(frame, winStride=winStride, padding=(4,4)
                                            , scale=scale)

    # apply nms
    rects = nms(rects, probs=None, overlapThresh=0.65)

    return rects


def intersection_over_union(boxA, boxB):

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
    ap.add_argument("-af", "--annotationsFile", required=False,
                    default="detectorTesting/videoframe_annotations.pickle")
    ap.add_argument("-v", "--video", required=False,
                    default="testVideo/IMG_0330.MOV")
    ap.add_argument("-d", "--detection", required=False,
                    default="dNN")
    ap.add_argument("-V", "--verbose", type=bool, required=False,
                    default=False)
    args = vars(ap.parse_args())

    # Initialize variables for metrics
    IoUTotalAvg = 0
    totalFalsePositives = 0
    totalFalseNegatives = 0
    totalTruePositives = 0
    totalAccuracy = 0
    precision = []
    recall = []

    # HOG detection parameters
    winStride = (4, 4)
    scale = 2

    # iteration variables
    n = 0
    anno = 0
    frameCount = 0

    # list of objects being tracked
    trackers = []

    # load annotations into list
    with open(args["annotationsFile"], 'rb') as annotations_list_file:
        annotations_list = pickle.load(annotations_list_file)

    vs = cv2.VideoCapture(args["video"])

    videoFPS = vs.get(cv2.CAP_PROP_FPS)
    print("Video stream FPS: {:.2f}".format(videoFPS))

    # start the frames per second throughput estimator
    fps = FPS().start()

    while True:

        grabbed, frame = vs.read()

        if frame is None:
            break

        if anno < len(annotations_list) - 1:
            bbox = annotations_list[n]

        else:
            break

        n += 1

        IoU = []
        iouTotal = 0

        frame = imutils.resize(frame, width=int(frame.shape[0]/1))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gtbbox = bbox
        predbbox = []

        for gtbox in gtbbox:

            gtX1 = gtbox[0]
            gtY1 = gtbox[1]
            gtX2 = gtbox[0] + gtbox[2]
            gtY2 = gtbox[1] + gtbox[3]

            cv2.rectangle(frame, (gtX1, gtY1), (gtX2, gtY2), (0, 255, 0), 2)

        if len(trackers) == 0 or frameCount == 20:

            # reset frame count
            frameCount = 0

            # clear tracked items
            trackers.clear()

            if args["detection"] == "dNN":
                # generate detection box
                predbbox = dnnDetection(frame)

            else:
                predbbox = hogDetection(frame, winStride, scale)

            for pred in predbbox:
                # create instance of tracker
                t = dlib.correlation_tracker()
                rect = dlib.rectangle(pred[0], pred[1], (pred[2] + pred[0]), (pred[3] + pred[1]))
                t.start_track(rgb, rect)

                trackers.append(t)

                cv2.rectangle(frame, (pred[0], pred[1]),
                              ((pred[2] + pred[0]), (pred[3] + pred[1])), (0, 0, 255), 2)

        # detection already performed so track objects detected
        else:
            # loop over each of the tracked objects
            for t in trackers:
                # update the position of the tracked object
                t.update(rgb)
                pos = t.get_position()

                x1 = int(pos.left())
                y1 = int(pos.top())
                x2 = int(pos.right())
                y2 = int(pos.bottom())

                rect = (x1, y1, x2 - x1, y2 - y1)
                predbbox.append(rect)

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)

        for i in gtbbox:
            iouMax = 0

            for j in predbbox:
                iou = intersection_over_union((i[0], i[1], i[2] + i[0], i[3] + i[1]),
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
            iouAvg = iouTotal / len(IoU)

        else:
            iouAvg = 0

        # cv2.putText(im, "IoU: {:.4f}".format(iou), (10, 30),
        #          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        truePositives = len(IoU)
        falseNegatives = len(gtbbox) - len(IoU)

        if args["verbose"]:

            print("Number of false Positives: {}".format(falsePositives))
            print("Number of false Negatives: {}".format(falseNegatives))
            print("Number of true Positives: {}".format(truePositives))
            print("IoU's : {}".format(IoU))
            print("Avg IoU: {}".format(iouAvg))

        IoUTotalAvg += iouAvg
        totalFalseNegatives += falseNegatives
        totalFalsePositives += falsePositives
        totalTruePositives += truePositives
        totalAccuracy += truePositives / len(gtbbox)

        if truePositives != 0 or falsePositives != 0:
            precision.append(truePositives / (truePositives + falsePositives))
        else:
            precision.append(0)

        if truePositives != 0 or falseNegatives != 0:
            recall.append(truePositives / (truePositives + falseNegatives))
        else:
            recall.append(0)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        anno += 1
        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()

    IoUAverage = IoUTotalAvg / n
    avgFalseNegatives = totalFalseNegatives / n
    avgFalsePositives = totalFalsePositives / n
    avgTruePositives = totalTruePositives / n
    avgAccuracy = totalAccuracy / n

    print("\nIoU average: {}".format(IoUAverage))
    print("average false negatives: {}".format(avgFalseNegatives))
    print("average false positives: {}".format(avgFalsePositives))
    print("average true positives: {}".format(avgTruePositives))
    print("Accuracy: {}".format(avgAccuracy))
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    performanceDrop = ((videoFPS - fps.fps()) / videoFPS) * 100

    print("FPS performance drop: {:.2f} %".format(performanceDrop))
    # print("Precision: {}".format(precision))
    # print("Recall: {}".format(recall))

    plt.plot(recall, precision, 'bo')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
