# FYP_Human_Detection_Engine
This Repository contains the source code for my Final Year Project which is a Human Detection Engine for video. 
The engine can be run two ways one with the Movidius Neural Compute Stick and one without.

## Instructions for running without Movidius NCS:
To run the appilication without the Movidius NCS run the dnnDetector_and _Tracker.py script in an environment
with the following packages installed:
* Python 3
* OpenCV 3.4.5
* imutils
* dlib

## Instructions for running with Movidius NCS:
To run the appilication with the Movidius NCS run the ncs_detector_and _tracker.py script in an environment
with the following packages installed:
* Python 3
* Movidius ncsdk API found here: https://github.com/movidius/ncsdk
* OpenCV 3.4.5
* imutils
* dlib

