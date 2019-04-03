from mvnc import mvncapi as mvnc
import dlib
import imutils
from imutils.video import FPS
import argparse
import numpy as np
import time
import cv2

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# frame dimensions should be square
processing_dims = (300, 300)
disp_dims = (700, 700)
# calculate the multiplier needed to scale the bboxes
disp_factor = int(disp_dims[0] / processing_dims[0])


def pre_process_image(image):
    # pre-process image
    pproc = cv2.resize(image, processing_dims)
    pproc = pproc - 127.5
    pproc = pproc * 0.007843
    pproc = pproc.astype(np.float16)

    return pproc


def predict(image, graph):
    image = pre_process_image(image)

    print("running inference")
    # send image to the NCS and run a forward pass to grab
    # network predictions
    graph.LoadTensor(image, None)
    (output, _) = graph.GetResult()

    # grab the number of valid object predictions from the output,
    # and initialize the list of predictions
    num_valid_boxes = output[0]
    pred = []

    # loop over results
    for box_index in range(int(num_valid_boxes)):
        # calculate base index so that bounding box information can be extracted
        base_index = 7 + box_index * 7

        if (not np.isfinite(output[base_index]) or
                not np.isfinite(output[base_index + 1]) or
                not np.isfinite(output[base_index + 2]) or
                not np.isfinite(output[base_index + 3]) or
                not np.isfinite(output[base_index + 4]) or
                not np.isfinite(output[base_index + 5]) or
                not np.isfinite(output[base_index + 6])):
            continue

        # move on if object is not a person
        if CLASSES[int(output[base_index + 1])] != "person":
                continue
            # extract the image width and height and clip the boxes to the
            # image size in case network returns boxes outside of the image
            # boundaries
        (h, w) = image.shape[:2]
        x1 = max(0, int(output[base_index + 3] * w))
        y1 = max(0, int(output[base_index + 4] * h))
        x2 = min(w, int(output[base_index + 5] * w))
        y2 = min(h, int(output[base_index + 6] * h))

        # grab the prediction class label, confidence (i.e., probability),
        # and bounding box (x, y)-coordinates
        pred_class = int(output[base_index + 1])
        pred_conf = output[base_index + 2]
        pred_boxpts = ((x1, y1), (x2, y2))

        # create prediction tuple and append the prediction to the
        # predictions list
        prediction = (pred_class, pred_conf, pred_boxpts)
        predictions.append(prediction)

        # return the list of predictions to the calling function
    return predictions


def initialize_device():
    # grab a list of all NCS devices plugged in to USB
    print("[INFO] finding NCS devices...")
    devices = mvnc.EnumerateDevices()

    # if no devices found, exit the script
    if len(devices) < 1:
        print("[INFO] No devices found. Please plug in a NCS")
        quit()

    # use the first device as only using one NCS
    print("[INFO] found {} devices. device0 will be used. "
          "opening device0...".format(len(devices)))
    device = mvnc.Device(devices[0])

    return device


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--graph", required=True,
                help="path to input graph file")
    ap.add_argument("-m", "--mode", required=False, default='test',
                    type=str)
    ap.add_argument("-v", "--video", default="video/IMG_0330.MOV")
    args = vars(ap.parse_args())

    mode = args["mode"]
    video = args["video"]

    # initialize the NCS Device
    device = initialize_device()

    # Open the device
    device.OpenDevice()

    # open the mobileNetSSD graph file
    print("loading the graph file into memory")
    with open(args["graph"], mode="rb") as f:
        graph_in_mem = f.read()

    # load the graph into the NCS
    print("allocating the graph on the NCS...")
    graph = device.AllocateGraph(graph_in_mem)

    # initialize the video stream if live load web cam
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

    # get video stream frame rate for bench marking
    videoFPS = vs.get(cv2.CAP_PROP_FPS)
    print("Video stream FPS: {:.2f}".format(videoFPS))

    # initialize the list for class labels
    labels = []

    # frame counter
    n = 0

    # start the frames per second throughput estimator
    fps = FPS().start()

    while True:
        try:
            # grab the frame from the threaded video stream
            # make a copy of the frame and resize it for display/video purposes
            grabbed, frame = vs.read()

            if frame is None:
                break

            image_for_disp = cv2.resize(frame, disp_dims)

            # get an rgb version of the frame and resize to processing dims for trackers
            rgb = cv2.cvtColor(image_for_disp, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, processing_dims)

            # if first frame or no items tracked or 20th frame run detection
            if n == 0 or n % 20 == 0:
                # initialize the list of object trackers
                trackers = []

                # calculate the inference time
                start = time.time()
                # use the NCS to acquire predictions
                predictions = predict(image_for_disp, graph)
                end = time.time()
                runtime = end - start
                print("inference on NCS works! and took : {:.2f} seconds".format(runtime))

                # loop over our predictions
                for (i, pred) in enumerate(predictions):
                    # extract prediction data
                    (pred_class, pred_conf, pred_boxpts) = pred

                    # filter out weak detections by ensuring the `confidence`
                    # is greater 0.5
                    if pred_conf > 0.5:
                        # build a label consisting of the predicted class and
                        # associated probability
                        label = "{}: {:.2f}%".format(CLASSES[pred_class],
                                                     pred_conf * 100)

                        # extract information from the prediction boxpoints
                        (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])

                        t = dlib.correlation_tracker()
                        rect = dlib.rectangle(ptA[0], ptA[1], ptB[0], ptB[1])
                        t.start_track(rgb, rect)

                        labels.append(label)
                        trackers.append(t)

                        # Scale bounding box coordinates up to display image size
                        ptA = (ptA[0] * disp_factor, ptA[1] * disp_factor)
                        ptB = (ptB[0] * disp_factor, ptB[1] * disp_factor)
                        (startX, startY) = (ptA[0], ptA[1])

                        # place label on top left of bounding box
                        y = startY - 15 if startY - 15 > 15 else startY + 15

                        # display the rectangle and label text
                        cv2.rectangle(image_for_disp, ptA, ptB,
                                      (0, 255, 0), 2)

                        cv2.putText(image_for_disp, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                # loop over each of the trackers
                for (t, l) in zip(trackers, labels):
                    # update the tracker and grab the position of the tracked object
                    t.update(rgb)
                    pos = t.get_position()

                    # unpack the position object
                    startX = int(pos.left()) * disp_factor
                    startY = int(pos.top()) * disp_factor
                    endX = int(pos.right()) * disp_factor
                    endY = int(pos.bottom()) * disp_factor

                    # draw the bounding box from the correlation object tracker
                    cv2.rectangle(image_for_disp, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
                    cv2.putText(image_for_disp, l, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            n += 1

            # display the frame to the screen
            cv2.imshow("Output", image_for_disp)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # update the FPS counter
            fps.update()

        # if "ctrl+c" is pressed in the terminal, break from the loop
        except KeyboardInterrupt:
            break

        # if there's a problem reading a frame, break gracefully
        except AttributeError:
            break

    # stop the timer and display FPS information
    fps.stop()
    print("elapsed time: {:.2f}".format(fps.elapsed()))
    print("approx. FPS: {:.2f}".format(fps.fps()))

    performanceDrop = ((videoFPS - fps.fps())/videoFPS) * 100

    print("FPS performance drop: {:.2f} %".format(performanceDrop))

    # clean up the graph and device
    graph.DeallocateGraph()
    device.CloseDevice()

    # cleanup video
    cv2.destroyAllWindows()
    vs.release()
