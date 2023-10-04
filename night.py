import cv2

def diffimage(t0, t1, t2):

    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

cam = cv2.VideoCapture(0)

t1 = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t2 = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

winname = "Night_vision"
cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)

while True:
    cv2.imshow(winname, diffimage(t1, t, t2))

    t1 = t 
    t = t2
    t2 = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

    if cv2.waitKey(10) == 27:
        cam.release()
        cv2.destroyAllWindows()
        break