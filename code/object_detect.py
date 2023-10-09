import cv2
import matplotlib.pyplot as plt

# reading config file for object detection
config_file = './Object_detection_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = './Object_detection_files/frozen_inference_graph.pb'

# intializing model to detect object
model = cv2.dnn_DetectionModel(frozen_model, config_file)

imgbackground = cv2.imread("./GUI Content/HACKTHON PROJECT.png") # Reading background image

# reading file that have the name of objects we can detect
classLabels = []
file_name = './Object_detection_files/labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
# print(classLabels) # printing the object names

# setting the frame size, scale and mean
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127, 5, 127.5))
model.setInputSwapRB(True)

def obj_detection(path):
    # Capturing the video
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Can't open the video")
    # setting font and font scale
    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN

    while True:
        # reading frames
        ret, frame = cap.read()

        ClassIndex, confidece, bbox = model.detect(frame, confThreshold = 0.5)
        print(ClassIndex)

        if(len(ClassIndex)!= 0):
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
                if ClassInd <= 80:
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, (0,255,0), 3)

        frame = cv2.resize(frame, (740, 480))
        imgbackground[150:150+480, 50:50+740] = frame

        cv2.putText(imgbackground, f"Press 'esc' to exit.", (880,540),cv2.FONT_HERSHEY_DUPLEX,0.6, (0,0,0), 1)

        cv2.imshow("Obj Detection", imgbackground)

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = input("Enter the path for prerecorded video or '0' for webcam: ")
    obj_detection(0) if path == "0" else obj_detection(path)

