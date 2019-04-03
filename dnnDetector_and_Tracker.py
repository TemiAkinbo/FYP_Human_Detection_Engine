from imutils.video import FPS
import numpy as np
import imutils
import dlib
import cv2
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scale", required=False, help="video frame scale factor",
                    default=2.25, type=float)
parser.add_argument("-m", "--mode", required=False, default='test',
                    type=str)
args = parser.parse_args()

resize_factor = args.scale
mode = args.mode

prototxt = "model/MobileNetSSD_deploy.prototxt"
caffeModel = "model/MobileNetSSD_deploy.caffemodel"
video = "testVideo/IMG_0330.MOV"


# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# load mobilenetSSD Caffe model from disk
print("loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, caffeModel)

# initialize the video stream
if mode == 'live':
    print("starting live webcam video stream...")
    vs = cv2.VideoCapture(0)
    time.sleep(1)
    grabbed, frame = vs.read()

    if frame is None:
        print("failed to start webcam video stream using testVideo instead")
        vs = cv2.VideoCapture(video)

else:
    print("starting test video stream...")
    vs = cv2.VideoCapture(video)

videoFPS = vs.get(cv2.CAP_PROP_FPS)
print("Video stream FPS: {:.2f}".format(videoFPS))

# calculating resize width
grabbed, frame = vs.read()
original_dim = frame.shape

print("Video Stream Original Dimensions: {}".format(original_dim))

resize = int(original_dim[0]/resize_factor)

# initialize the list of object trackers and class labels
labels = []

# frame count
n = 0

# start the timer
fps = FPS().start()

while True:
    # grab the next frame from the video file
    (grabbed, frame) = vs.read()

    if frame is None:
        break

    image_to_display = imutils.resize(frame, width=int(frame.shape[0]/1))
    # resize frame to scale factor
    frame = imutils.resize(frame, width=resize)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if frame count is 0 or is the 20th frame perform object detection
    if n == 0 or n % 20 == 0:

        # clear current trackers
        trackers = []

        # get height and width of frame to be used as arguments for pre-processing function
        (h, w) = frame.shape[:2]

        # image pre processing before passing to DNN model
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

        # pass image to network input
        net.setInput(blob)

        # calculate inference time and run inference
        start = time.time()
        detections = net.forward()
        end = time.time()
        runtime = end - start
        print("detection runtime: {:.2f}".format(runtime))
        print("detection array type : {} & shape : {}".format(type(detections),detections.shape))

        # for each detection of class person with confidence > 0.5 add to object tracked list
        # and draw bounding box
        for i in np.arange(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            # get the index of the label of the detected object
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            if CLASSES[idx] is not "person":
                continue

            if confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # rescale detection box size to frame size
                (startX, startY, endX, endY) = box.astype("int")

                t = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                t.start_track(rgb, rect)

                labels.append(label)
                trackers.append(t)

                cv2.rectangle(image_to_display, (int(startX * resize_factor), int(startY * resize_factor)),
                              (int(endX * resize_factor), int(endY * resize_factor)), (0, 255, 0), 2)
                cv2.putText(image_to_display, label, (int(startX * resize_factor), int((startY * resize_factor) - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # otherwise detection already performed so track
    # objects in trackers list
    else:
        # loop over each of the trackers
        for (t, l) in zip(trackers, labels):
            # update the tracker and grab the position of the tracked
            # object
            t.update(rgb)
            pos = t.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            cv2.rectangle(image_to_display, (int(startX * resize_factor), int(startY * resize_factor)),
                          (int(endX * resize_factor), int(endY * resize_factor)), (0, 255, 0), 2)
            cv2.putText(image_to_display, label, (int(startX * resize_factor), int((startY * resize_factor) - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # increment frame number
    n += 1

    # image_to_display = imutils.resize(image_to_display, width=720)

    # show the output frame
    cv2.imshow("Frame", image_to_display)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("elapsed time: {:.2f}".format(fps.elapsed()))
print("approx. FPS: {:.2f}".format(fps.fps()))

performanceDrop = ((videoFPS - fps.fps())/videoFPS) * 100

print("FPS performance drop: {:.2f} %".format(performanceDrop))


# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
