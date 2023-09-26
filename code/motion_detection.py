import cv2
import imutils
import threading
import winsound

def motion_tracking():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 740)

    _, start_frame = cap.read()
    start_frame = imutils.resize(start_frame, width=500)
    start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
    start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)

    imgbackground = cv2.imread("./GUI Content/HACKTHON PROJECT.png")

    alarm = False
    alarm_mode = False
    alarm_counter = 0

    def beep_alarm():
        global alarm
        for _ in range(5):
            if not alarm_mode:
                break
            # print("ALARM")
            winsound.Beep(2500, 1000)
        alarm = False

    while True:
        _, frame = cap.read()
        frame = imutils.resize(frame, width=500)

        if alarm_mode:
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)

            difference = cv2.absdiff(frame_bw, start_frame)
            threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
            start_frame = frame_bw

            if threshold.sum() > 250 :
                # print(threshold.sum())
                # print(alarm_counter)
                alarm_counter += 1
            else:
                alarm_counter -= 1 

            threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)

            threshold = cv2.resize(threshold, (740, 480))
            imgbackground[150:150+480, 50:50+740] = threshold
        else:
            frame = cv2.resize(frame, (740, 480))
            imgbackground[150:150+480, 50:50+740] = frame

        cv2.putText(imgbackground, f"Press 'c' to start motion detection.", (880,520),cv2.FONT_HERSHEY_DUPLEX,0.6, (0,0,0), 1)
        cv2.putText(imgbackground, f"Press 'a' to exit.", (880,540),cv2.FONT_HERSHEY_DUPLEX,0.6, (0,0,0), 1)
        cv2.imshow("Motion_Tracking", imgbackground)

        if alarm_counter > 5:
            if not alarm:
                alarm = True
                threading.Thread(target=beep_alarm).start()

        key_pressed = cv2.waitKey(30)
        if key_pressed == ord("c"):
            alarm_mode = not alarm_mode
            alarm_counter = 0
        if key_pressed == ord("a"):
            alarm_mode = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    motion_tracking()
    
