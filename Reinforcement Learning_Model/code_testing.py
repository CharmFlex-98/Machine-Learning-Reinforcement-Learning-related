import cv2
import random
import time
import zmq
import base64
import numpy as np


def find_object(_outputs, _frame):
    bboxes = []
    class_ids = []
    confidences = []
    frame_h, frame_w = _frame.shape[:2]
    for output in _outputs:  # each 3 different scales
        for detection in output:
            scores = detection[5:]
            detected_classID = scores.argmax(0)
            confidence = scores[detected_classID]
            if confidence > confidence_threshold:
                w, h = int(detection[2] * frame_w), int(detection[3] * frame_h)
                x, y = int(detection[0] * frame_w - w / 2), int(detection[1] * frame_h - h / 2)  # upper left coordinate
                bboxes.append([x, y, w, h])
                class_ids.append(detected_classID)
                confidences.append(float(confidence))

    object_index = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
    for i in object_index:
        i = i[0]
        box = bboxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        collect_true_sample(class_names[class_ids[i]], [x, y, w, h])
        if class_names[class_ids[i]] == 'Human':
            continue
        cv2.rectangle(_frame, (x, y), (x + w, y + h), class_colour_dict[class_names[class_ids[i]]], 2)
        cv2.putText(_frame, '{} : {}'.format(class_names[class_ids[i]], round(confidences[i] * 100, 1)), (x, y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, class_colour_dict[class_names[class_ids[i]]], 2)


def collect_true_sample(class_name, coordinate):
    if class_name == 'Human':
        person.append(coordinate)
    elif class_name == 'Helmet':
        cell_phone.append(coordinate)
    else:
        pass


def checking(_frame):
    for p in person:
        equipped = False
        for index, item in enumerate(cell_phone):
            if p[0] <= item[0] + item[2] / 2 <= p[0] + p[2] and p[1] <= item[1] + item[3] / 2 <= p[1] + p[3]:
                cv2.rectangle(_frame, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), [255, 0, 255], 2)
                cv2.putText(_frame, '{}'.format('good!'), (p[0], p[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 255])
                equipped = True
                break
        if not equipped:
            cv2.rectangle(_frame, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), [255, 255, 255], 2)
            cv2.putText(_frame, '{}'.format('bad!'), (p[0], p[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                        [255, 255, 255])


def class_colour(classes):
    for _class in classes:
        R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        colour = (R, G, B)
        class_colour_dict[_class] = colour


cam = True
server = False
image_path = r'C:\Users\ASUS\PycharmProjects\Pytorch\Yolov3\cheongbakProject\01905.jpg'
names_path = r'C:\Users\ASUS\darknet\build\darknet\x64\data\coco.names'
cfg_path = r'C:\Users\ASUS\darknet\build\darknet\x64\cfg\yolov4-tiny.cfg'
weights_path = r'C:\Users\ASUS\darknet\build\darknet\x64\weights\yolov4-tiny.weights'
width = 416
height = 416
confidence_threshold = 0.3
nms_threshold = 0.3

class_colour_dict = {}
with open(names_path, 'r') as file:
    class_names = file.read().rstrip('\n').split('\n')
    class_colour(class_names)

model = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

if cam:
    if not server:
        camera = cv2.VideoCapture(0)
        camera.set(3, 500)
        camera.set(4, 500)
    else:
        """实例化用来接收帧的zmq对象"""
        context = zmq.Context()
        """zmq对象建立TCP链接"""
        footage_socket = context.socket(zmq.REQ)
        footage_socket.bind('tcp://192.168.0.15:5555')

    counter = 0
    fps=[]
    while True:
        start_time = time.time()
        if not server:
            success, frame = camera.read()
        else:
            footage_socket.send_string('please send a frame')
            frame = footage_socket.recv()  # 接收TCP传输过来的一帧视频图像数据
            img = base64.b64decode(frame)  # 把数据进行base64解码后储存到内存img变量中
            npimg = np.frombuffer(img, dtype=np.uint8)  # 把这段缓存解码成一维数组
            frame = cv2.imdecode(npimg, 1)  # 将一维数组解码为图像source

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (width, height), [0, 0, 0], crop=False)
        model.setInput(blob)

        layers = model.getLayerNames()
        outputNames = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]
        outputs = model.forward(outputNames)  # these are the outputs with results

        person = []
        cell_phone = []

        find_object(outputs, frame)
        checking(frame)

        cv2.imshow('cam', frame)
        cv2.waitKey(1)
        print(round(1 / (time.time() - start_time), 1))

else:
    image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (width, height), [0, 0, 0], crop=False)
    model.setInput(blob)

    layers = model.getLayerNames()
    outputNames = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    outputs = model.forward(outputNames)  # these are the outputs with results

    person = []
    cell_phone = []

    find_object(outputs, image)
    checking(image)
    cv2.imwrite('opencv2.jpg', image)

