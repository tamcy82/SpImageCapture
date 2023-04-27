# importing the python open cv library
import json
import os
import sys
import cv2
import ctypes
from time import sleep
from datetime import date
import re

import numpy as np


# Handler app message
def message_handler(msg, code=None):
    level = 1
    info = {'code': code, 'msg': msg}
    if level == 1:
        print(msg)
    else:
        if code is not None:
            # json encode the info
            json_info = json.dumps(info)
            # print the json info
            print(json_info)


# Function to get imaging path
def get_imaging_path(study):
    if study == '':
        return False
    export_paths = [
        '\\\\ctc-network.intranet\\dfs\\BIOTR\\01 Ongoing Studies\\',
        '\\\\ctc-network.intranet\\dfs\\BIOTR\\02 Closed Studies\\'
    ]
    imaging_dir = "08 Specimen Imaging"
    study = study.upper()
    # Regex to check study input
    if not re.match(r'^\d{4}[A-Z]{3}\d{1}$', study):
        message_handler('Invalid study')
        return False
    message_handler("Loading Study")
    # Extract digits of CTC No.
    ctc_no_digit = ''
    # Extract first digits
    for c in study:
        if c.isdigit():
            ctc_no_digit += c
        else:
            break
    # Find study dir
    study_dir = ''
    # Loop through all paths
    for p in export_paths:
        # List directories
        for dir in os.listdir(p):
            # Check if study dir exists
            if dir.startswith(ctc_no_digit):
                study_dir = p + dir
                break
        if study_dir != '':
            break
    if study_dir == '':
        message_handler('Study not found')
        return False
    # Find imaging dir
    imaging_path = os.path.join(study_dir, imaging_dir)
    message_handler("Checking imaging path: " + imaging_path)
    if not os.path.exists(imaging_path):
        # Create imaging dir
        if not os.mkdir(imaging_path):
            message_handler('Failed to create imaging directory', 101)
            return False
    return imaging_path


# main function
def main():
    # Basic setup for the app
    # Imaging path
    imaging_path = False
    # Default study
    study = ""
    # Get command line arguments
    if len(sys.argv) > 1:
        study = sys.argv[1]
        study.upper()
        imaging_path = get_imaging_path(study)
        if not imaging_path:
            message_handler('Invalid study', 102)
            sleep(10)
            return
    else:
        while not imaging_path:
            study = input('Enter study: ')
            study.upper()
            imaging_path = get_imaging_path(study)

    # Generate file name
    def generate_file_path(ext='jpg'):
        # The counter for the image file
        img_counter = 0
        # The prefix for the image file
        img_name_prefix = study.upper() + '_' + date.today().strftime("%Y%m%d") + '_Imaging_'
        # The full file name
        img_name = img_name_prefix + f'{img_counter}.' + ext
        while os.path.exists(os.path.join(imaging_path, img_name)):
            img_counter += 1
            img_name = img_name_prefix + f'{img_counter}.' + ext
        img_counter += 1
        return os.path.join(imaging_path, img_name)

    # list available cameras
    camera_count = 0
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.read()[0]:
            camera_count += 1
        cap.release()
    if camera_count == 0:
        message_handler('No camera detected', 103)
        # Sleep
        sleep(5)
        return
    print("No. of available cameras: " + str(camera_count))
    cam_port = -1
    # Select camera
    while True:
        cam_port = input('Select camera [0]: ')
        if cam_port == '':
            cam_port = 0
            break
        if cam_port.lstrip('-').isdigit():
            cam_port = int(cam_port)
            break
        print('Invalid input')
    # Alternative
    # Select camera
    # while cam_port > camera_count or cam_port < 0:
    #     cam_port = input('Select camera [0]: ')
    #     if cam_port == '':
    #         cam_port = 0
    #         break
    #     if not cam_port.isdigit():
    #         cam_port = -1
    #     cam_port = int(cam_port)
    # initialize the webcam
    cam = cv2.VideoCapture(cam_port, cv2.CAP_ANY)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cam.isOpened():
        message_handler('Cannot open camera', 103)
        # Sleep
        sleep(5)
        return
    message_handler("Preparing camera ...")
    # title of the app
    cv2.namedWindow('Specimen Imaging')
    # while loop
    while True:
        # Initializing the frame, ret
        ret, frame = cam.read()
        # if statement
        if not ret:
            message_handler('Failed to grab frame', 104)
            break
        # Display the frame
        cv2.imshow('Specimen Imaging', frame)
        # display frame info for debugging
        print(frame)
        cv2.setWindowProperty('Specimen Imaging', cv2.WND_PROP_TOPMOST, 1)
        # Wait
        k = cv2.waitKey(1)
        # if the escape key has been pressed, the app will stop
        if k % 256 == 27:
            message_handler('Escape hit, closing the app')
            break
        # if the spacebar key or enter key has been pressed
        elif k % 256 == 32 or k % 256 == 13:
            # screenshots will be taken
            # cv2.imwrite(os.path.join(imaging_path, img_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(new_file, frame, [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])
            new_file = generate_file_path('jpg')
            cv2.imwrite(new_file, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
            message_handler('Screenshot taken: ' + new_file)
            # Show a popup message box
            ctypes.windll.user32.MessageBoxW(0, "Screenshot taken", "Specimen Imaging", 0x1000)
        # if window is closed
        if cv2.getWindowProperty('Specimen Imaging', cv2.WND_PROP_VISIBLE) < 1:
            message_handler('Closing the camera')
            break
    # release the camera
    cam.release()
    # stops the camera window
    cam.destroyAllWindows()
    # Completed
    message_handler('Completed', 100)


# main function 2
def main_2():
    # Basic setup for the app
    # Imaging path
    imaging_path = False
    # Default study
    study = ""
    # Get command line arguments
    if len(sys.argv) > 1:
        study = sys.argv[1]
        study.upper()
        imaging_path = get_imaging_path(study)
        if not imaging_path:
            message_handler('Invalid study', 102)
            sleep(10)
            return
    else:
        while not imaging_path:
            study = input('Enter study: ')
            study.upper()
            imaging_path = get_imaging_path(study)

    # Generate file name
    def generate_file_path(ext='tiff'):
        # The counter for the image file
        img_counter = 0
        # The prefix for the image file
        img_name_prefix = study.upper() + '_' + date.today().strftime("%Y%m%d") + '_Imaging_'
        # The full file name
        img_name = img_name_prefix + f'{img_counter}.' + ext
        while os.path.exists(os.path.join(imaging_path, img_name)):
            img_counter += 1
            img_name = img_name_prefix + f'{img_counter}.' + ext
        img_counter += 1
        return os.path.join(imaging_path, img_name)

    # import coco.names
    print("Loading object detection model ...")
    classes = []
    with open('./yolo/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    # import yolov3.cfg and yolov3.weights
    net = cv2.dnn.readNetFromDarknet('./yolo/yolov3.cfg', './yolo/yolov3.weights')
    model = cv2.dnn_DetectionModel(net)
    # set the input params
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
    # get the output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # generate different colors for different classes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # list available cameras
    camera_count = 0
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.read()[0]:
            camera_count += 1
        cap.release()
    print("No. of available cameras: " + str(camera_count))
    cam_port = -1
    # Select camera
    while cam_port > camera_count or cam_port < 0:
        cam_port = input('Select camera [0]: ')
        if cam_port == '':
            cam_port = 0
            break
        if not cam_port.isdigit():
            cam_port = -1
        cam_port = int(cam_port)
    # initialize the webcam
    cam = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cam.isOpened():
        message_handler('Cannot open camera', 103)
        # Sleep
        sleep(5)
        return
    message_handler("Preparing camera ...")
    # title of the app
    cv2.namedWindow('Specimen Imaging')
    # while loop
    while True:
        # Initializing the frame, ret
        ret, frame = cam.read()
        # if statement
        if not ret:
            message_handler('Failed to grab frame', 104)
            break
        # classIds, scores, boxes = model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)
        # for classId, score, box in zip(classIds, scores, boxes):
        #     cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
        #     cv2.putText(frame, classes[classId - 1], (box[0] + 10, box[1] + 30),
        #                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        #     cv2.putText(frame, str(round(score * 100, 2)), (box[0] + 200, box[1] + 30),
        #                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
        if len(indexes) == 0:
            cv2.putText(frame, 'No Object Detected', (10, 50), font, 3, (0, 0, 255), 3)
        # Display the frame
        cv2.imshow('Specimen Imaging', frame)
        # cv2.setWindowProperty('Specimen Imaging', cv2.WND_PROP_TOPMOST, 1)
        # wait for 1ms
        k = cv2.waitKey(1)
        # if the escape key has been pressed, the app will stop
        if k % 256 == 27:
            message_handler('Escape hit, closing the app')
            break
        # if the spacebar key or enter key has been pressed
        elif k % 256 == 32 or k % 256 == 13:
            # screenshots will be taken
            # cv2.imwrite(os.path.join(imaging_path, img_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            new_file = generate_file_path('tiff')
            _, frame2 = cam.read()
            cv2.imwrite(new_file, frame2, [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])
            message_handler('Screenshot taken: ' + new_file)
            # Show a popup message box
            ctypes.windll.user32.MessageBoxW(0, "Screenshot taken", "Specimen Imaging", 0)
        # if window is closed
        if cv2.getWindowProperty('Specimen Imaging', cv2.WND_PROP_VISIBLE) < 1:
            message_handler('Closing the camera')
            break
    # release the camera
    cam.release()
    # stops the camera window
    cam.destroyAllWindows()
    # Completed
    message_handler('Completed', 100)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
