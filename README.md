# moving-objects-using-openCV
Objectives
• Design a Python program that extracts and counts moving objects, e.g. people, cars and others using
background modelling and detects pedestrians using pre-trained MobileNet SSD object detector.
Introduction
Extraction of moving objects and detection of pedestrians from a sequence of images or video is often used
in many video analysis tasks. For instance, it is a key component in intelligent video surveillance systems
and autonomous driving. In this assignment, you are required to develop a program in Python using
OpenCV 4.6.0 to detect, separate and count moving objects from a given sequence of images or video
captured by a stationary camera and to detect pedestrians. There are two tasks.
Task One (15%) – Background modelling
In this task, you are required to extract moving objects using Gaussian Mixture background modelling.
There are three key steps involved in the extracting and counting moving objects:
1. Resize the video frame to a size comparable to VGA
2. Detecting moving pixels using background modelling and subtraction,
3. Removing noisy detection using morphological operators or majority voting and
4. Count separate moving objects using connected component analysis.
5. Classify each object (or connected component) into person, car and other by simply using the ratio of
width and height of the connected components.
OpenCV 4.6.0 provides various algorithms for each of the steps 1 to 3. However, you MAY have to
implement your own connected component analysis algorithm and classification algorithm. For simplicity,
you can assume that each connected component corresponds to one object.
Original Video Frame Estimated
Background Frame
Detected Moving
Pixels before
Filtering (in Binary
Mask)
Detected Objects
(in Original color)
SCIT, University of Wollongong
2 of 3
When running, the program should display the original video frame, estimated background frame, detected
moving pixels after the background modeling and subtraction (before any noise removal) and the detected
moving objects in a single window as illustrated above. The detected moving pixels before filtering should
be displayed in black and white (binary mask). The detected objects have to be displayed in its original RGB
color (all background pixels should be displayed in black). At the same time, the number of objects or
connected components should be output to the command window as illustrated below:
Frame 0001: 0 objects
Frame 0002: 0 objects
…
Frame 0031: 5 objects (2 persons, 1 car and 2 others)
Frame 0032: 6 objects (3 persons, 1 cars and 2 others)
…
Frame 1000: 10 objects (
…
Task Two (10%) – Detection and Tracking of Pedestrians
In this task, you are required to detect pedestrians (i.e. persons) using a OpenCV Deep Neural Network
(DNN) module and a MobileNet SSD detector pre-trained on the MS COCO dataset, to track and display the
detected pedestrians by providing same labels, i.e. 1, 2,
⋯
, 𝑛, to the same pedestrians across over times and
to select up to (3) pedestrians that are most close in space to the camera. Solutions to the tracking and
selection of up to three (3) most close pedestrians must be described in the comments at the beginning of
your Python code.
Note that
(a) the pre-trained MobileNet SSD is provided:
• model configuration – ssd
_mobilenet_v2_coco_2018_03_29.pbtxt.txt
• weights – frozen_inference_graph.pb
• names of object classes in MS COCO - object_detection_classes_coco.txt
(b) the size of the input images to the model is 300 × 300
(c) a quick tutorial on how to use the OpenCV DNN module and the pre-trained MobileNet SSD is
available at https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/.
(d) The pre-trained MobileNet SSD model is able to detect 80 classes of objects. This assignment is only
interested in pedestrians, i.e. persons.
Original video frame
Video frame with
overlapped
bounding-boxes of
detected pedestrians
Video frame with
detected and tracked
(i.e. labelled)
bounding-boxes
Video Frame with up to
3 detected and tracked
pedestrians that are
most closed to the
camera
SCIT, University of Wollongong
3 of 3
The program should display the original video frame, video frame with overlapped bounding-boxes of the
detected pedestrians, video frame with detected and tracked bounding-boxes and video frame with up to
three (3) closest objects to the camera. Display must be in a single window as illustrated above.
Requirements on coding
1. The program should be named as “movingObj” and shall take an option, either –b or -d and a
video filename as the input, e.g. movingObj –b videofile or movingObj –d
videofile. When –b is given, the program should perform Task One and when –d is given, the
program performs Task Two.
2. No other third-party libraries should be used in the program except OpenCV 4.6.0-Python (assuming
that numpy and matplotlib packages exist). The code has to be in Python.
3. Place your implementation in a single.py file called movingObj.py
4. Description of the solution to Task Two should be given as comments at the beginning of the
movingObj.py
Marking Scheme
Zero marks may be graded if your code cannot run or the code does not meet the requirements
Task One
1. Program structure, comments and usability (1%)
2. Read and display of the video frames (2%)
3. Background modeling or estimation (2%)
4. Subtraction and display of the detected moving pixels (2%)
5. Remove of noisy or false detection (2%)
6. Connected component analysis and display of the moving objects (2%)
7. Classification of moving objects (2%)
8. Output number of objects or connected components (2%)
Task Two
9. Description of the solution (1%)
10. Display of the detection results (in bounding-boxes) (3%)
11. Display of tracking results (3%)
12. Display of up to three tracked pedestrians that are most closed to the camera (3%)
Submission
Zip the movingObj.py file to your_login_name.zip. The zip file must be submitted via Moodle.
IMPORTANT:
a) DO NOT include and submit any videos and the tr-trained MobileNet SSD model in the zip file. Your
submission may not be accepted if you do so.
b) Submission through email WILL NOT be accepted
