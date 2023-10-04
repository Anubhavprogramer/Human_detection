import cv2
import os

def calculate_distance(known_height, face_height, focal_length):
    distance = (known_height * focal_length) / face_height
    return distance

def motion(path):
    cap = cv2.VideoCapture(path) #capturing video from webcam

    # Set the known height of a human face in meters
    known_face_height = 0.15  # Assuming an average face height of 15 centimeters (0.15 meters)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    
    # modes folder is opening here   
    foldermodepath = './GUI Content/modes'  # Use forward slash (/) for paths
    modePathList = os.listdir(foldermodepath)
    imgmodelist = []
    modechanger = 0 #helps to switch between modes
    # here we are creating mode list to show modes to user
    for mode_file in modePathList:
        imgmodelist.append(cv2.imread(os.path.join(foldermodepath, mode_file)))

    imgbackground = cv2.imread("./GUI Content/HACKTHON PROJECT.png") # Reading background image

    while cap.isOpened():

        diff = cv2.absdiff(frame1,frame2) # Checking the difference
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # Converting the image to gray scale image, so that it can detect properly
        blur = cv2.GaussianBlur(gray, (5,5), 0) # bluring the image to check threshold
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # setting threshold to detect
        dilate = cv2.dilate(thresh, None, iterations=3)  # 
        contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        distance = 0
        # for contour in contours:
        #     (x, y,  w, h) = cv2.boundingRect(contour)
        #     if cv2.contourArea(contour) < 800:
        #         continue
        #     cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #     focal_length = 400  # Replace with your actual focal length

        #     # Calculate the height of the detected face
        #     face_height = h

        #     # Calculate the distance in meters
        #     distance = calculate_distance(known_face_height, face_height, focal_length)

        cv2.drawContours(frame1, contours, -1, (0,255,0), 2) # Drawing detected area

        frame1 = cv2.resize(frame1, (740, 480))
        imgbackground[150:150+480, 50:50+740] = frame1
        imgmodelist[modechanger] = cv2.resize(imgmodelist[modechanger], (410, 383))
        imgbackground[45:45 + 383, 860:860 + 410, :] = imgmodelist[modechanger]  # Adding mode to the project

        frame1 = frame2

        ret, frame2 = cap.read()

        if cv2.waitKey(30) == 27:
            break

        # cv2.putText(frame1, f"Distance: {distance:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(imgbackground, f"Press 'esc' to exit.", (880,540),cv2.FONT_HERSHEY_DUPLEX,0.6, (0,0,0), 1)
        cv2.imshow("Motion_Tracking", imgbackground)

        imgmodelist[modechanger] = cv2.resize(imgmodelist[modechanger], (410, 383))
        imgbackground[45:45 + 383, 860:860 + 410, :] = imgmodelist[modechanger]  # Adding mode to the project
    cap.release()
    cv2.destroyAllWindows()

if __name__  == "__main__":
    path = input("Enter '0' for webcam and './Sample videos/sample video.avi' for prerecorded videos: ")
    motion(0) if path == "0" else motion(path)