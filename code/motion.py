import cv2

def motion():
    cap = cv2.VideoCapture("./Sample videos/sample video.avi") #capturing video from webcam

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    imgbackground = cv2.imread("./GUI Content/HACKTHON PROJECT.png") # Reading background image

    while cap.isOpened():

        diff = cv2.absdiff(frame1,frame2) # Checking the difference
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # Converting the image to gray scale image, so that it can detect properly
        blur = cv2.GaussianBlur(gray, (5,5), 0) # bluring the image to check threshold
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # setting threshold to detect
        dilate = cv2.dilate(thresh, None, iterations=3)  # 
        contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y,  w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) > 1000:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # cv2.drawContours(frame1, contours, -1, (0,255,0), 2) # Drawing detected area

        frame1 = cv2.resize(frame1, (740, 480))
        imgbackground[150:150+480, 50:50+740] = frame1

        frame1 = frame2

        ret, frame2 = cap.read()

        if cv2.waitKey(20) == ord("a"):
            break

        cv2.putText(imgbackground, f"Press 'a' to exit.", (880,540),cv2.FONT_HERSHEY_DUPLEX,0.6, (0,0,0), 1)
        cv2.imshow("Motion_Tracking", imgbackground)

    cap.release()
    cv2.destroyAllWindows()

if __name__  == "__main__":
    motion()