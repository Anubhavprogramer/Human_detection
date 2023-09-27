import os
import cv2
import pickle
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
from playsound import playsound # To playsounds
from motion import motion

cred = credentials.Certificate("ServiceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://humandetection-43fa7-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket':"humandetection-43fa7.appspot.com"
})

bucket = storage.bucket()

def calculate_distance(known_height, face_height, focal_length):
    distance = (known_height * focal_length) / face_height
    return distance

def detect(path):

    # Reading Harcascade file for Detection 
    full_body = cv2.CascadeClassifier("./Haarcascade files/haarcascade_fullbody.xml") # Reading for full body Detection 
    face = cv2.CascadeClassifier("./Haarcascade files/haarcascade_frontalface_default.xml") # Reading for face Detction 

    # Set the known height of a human face in meters
    known_face_height = 0.15  # Assuming an average face height of 15 centimeters (0.15 meters)


    # web cam setup
    cam = cv2.VideoCapture(path)
    cam.set(3, 740)
    cam.set(4, 480)

    cap = cv2.VideoCapture("./GUI Content/bottom_right.mp4")

    # modes folder is opening here   
    foldermodepath = './GUI Content/modes'  # Use forward slash (/) for paths
    modePathList = os.listdir(foldermodepath)
    imgmodelist = []
    # here we are creating mode list to show modes to user
    for mode_file in modePathList:
        imgmodelist.append(cv2.imread(os.path.join(foldermodepath, mode_file)))

    # assinging background image to project
    imgbackground = cv2.imread("./GUI Content/HACKTHON PROJECT.png")

    print("Loading the encoded file.....")
    # Load the encoding file
    with open("encodefile.p", "rb") as file:
        encodelistknownwithIDs = pickle.load(file)
    # splinting the encodelistknownwithIDs into id and encodedlistitem
    encodelistknown, personIDs = encodelistknownwithIDs
    print("Loaded the encoded file.....")

    # assing the initial values to cariables
    modechanger = 0 #helps to switch between modes
    PersonInfo=[]#pers's info will be stored here
    counter=0 
    id=-1 #stores the id to be shown
    imgperson=[] #stres image that have to be shown 
    
    while True:
        ret, img = cam.read()  # camera read here
        reg, bottom_right = cap.read()  # reading video to add texts
        if not ret:
            print("Error: Failed to capture frame.")
            break

        if not reg:
            print("Error: Failed to capture frame.")
            break

        col = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col = cv2.bilateralFilter(col,5,1,1)
        
        full_body_cap = full_body.detectMultiScale(
            col,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        face_cap = face.detectMultiScale(
            col,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        person = 1 # Variable to count person

        # Drawing full body regions in the frame
        for (x, y, w, h) in full_body_cap:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Calculate the focal length (you can measure this experimentally)
            focal_length = 600  # Replace with your actual focal length

            # Calculate the height of the detected face
            face_height = h

            # Calculate the distance in meters
            distance = calculate_distance(known_face_height, face_height, focal_length)

            person += 1  # incrimenting according to number of rectangles created

        # Drawing face regions in frame
        for (x, y, w, h) in face_cap:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            focal_length = 600  # Replace with your actual focal length

            # Calculate the height of the detected face
            face_height = h

            # Calculate the distance in meters
            distance = calculate_distance(known_face_height, face_height, focal_length)
            person += 1  # incrimenting according to number of rectangles created

        cv2.putText(img, f"Distance: {distance:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Displaying total number of persons in frame
        cv2.putText(bottom_right, f'Live Census: {person - 1} ', (40,50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 2)
        # Displaying info to exit the video window
        cv2.putText(bottom_right, f"Press 'a' to exit.", (40,175),cv2.FONT_HERSHEY_PLAIN,0.8, (0,0,0), 1)

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resizing the web cam output        
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # color change 

        # actual face recognition is started
        faceCurFrame = face_recognition.face_locations(imgS) 
        # Detecting the face
        enocdeCurrFrame = face_recognition.face_encodings(imgS, faceCurFrame)
        # actual face recognition is ended

        # Embiding the videos in Hackathon.png
        img = cv2.resize(img, (740, 480))
        bottom_right = cv2.resize(bottom_right, (420,210))
        imgbackground[150:150+480, 50:50+740] = img  # Adding webcam part to background
        imgbackground[440:440 + 210, 858:858 + 420] = bottom_right # Adding bottom right video to add text
        imgmodelist[modechanger] = cv2.resize(imgmodelist[modechanger], (410, 383))
        imgbackground[45:45 + 383, 860:860 + 410, :] = imgmodelist[modechanger]  # Adding mode to the project

        # unziping the data generated by face_recognition
        for encodeface, faceLoc in zip(enocdeCurrFrame, faceCurFrame):
            # matchning started
            matches = face_recognition.compare_faces(encodelistknown, encodeface)
            facedis = face_recognition.face_distance(encodelistknown, encodeface)
            # matching ended
            # print("matches", matches)
            # print("facedis", facedis)
            #here we are finding the minimum face distance

            matchIndex = np.argmin(facedis)

            # matchindex is the index which is detected
            # print("Match Index is: ",matchIndex)
            if matches[matchIndex]:
                # personIDs[matchIndex]

                # playsound("./Sound effects/Beep.mp3")  # Playing the sound

                # adding square to web cam  jo face detect krta hai
                # y1,x2,y2,x1 = faceLoc
                # y1,x2,y2,x1 = y1*4 ,x2*4 ,y2*4 ,x1*4  
                # bbox=60+x1,150+y1,x2-x1,y2-y1
                # imgbackground=cvzone.cornerRect(imgbackground,bbox,rt=0)
                
                # id jo images ko di hai vo ispe store hogi
                id = personIDs[matchIndex]
                # print(id)
                if counter == 0:
                    counter=1
                    modechanger=1
                    imgmodelist[modechanger] = cv2.resize(imgmodelist[modechanger], (410, 383))
                    imgbackground[45:45 + 383, 860:860 + 410, :] = imgmodelist[modechanger]

        if counter != 0:
            #yha first frame pe data download ho rha hai server se
            if counter ==1:
                # get the data
                PersonInfo = db.reference(f'data_of_people/{id}').get()
                # get the image from data base
                blob = bucket.get_blob(f'person/{id}.jpeg')
                array = np.frombuffer(blob.download_as_string(),np.uint8)
                imgperson = cv2.imdecode(array,cv2.COLOR_BGRA2BGR)
                # update data of the last time detected
                detectedtime=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                ref = db.reference(f'data_of_people/{id}')
                ref.child('LastDetectedTime').set(detectedtime)


            if counter<=30:
                # adding the data to output window
                # name
                (w,h),_=cv2.getTextSize(PersonInfo['name'],cv2.FONT_HERSHEY_COMPLEX,1,1)
                offset = (453-w)//2
                cv2.putText(imgbackground,str(PersonInfo['name']),(855+offset,100),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)

                # other field
                cv2.putText(imgbackground,str(PersonInfo['LastDetectedTime']),(1003,330),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,0),1)
                cv2.putText(imgbackground,str(PersonInfo['reason']),(988,365),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2)


                imgperson = cv2.resize(imgperson, (160, 160))
                imgbackground[110:110+160,993:993+160] = imgperson

            counter+=1
            #reseting the dependent vriables
            if counter>=30:
                counter=0
                modechanger=0
                PersonInfo=[]
                imgperson = []
                imgbackground[45:45 + 383, 860:860 + 410, :] = imgmodelist[modechanger]
        
        # yha pe  actual project show ho rha hai 
        cv2.imshow("Human_detect", imgbackground)
        #wait ki se image ko roka ja sakta hai 
        if cv2.waitKey(5) == ord("a"):
            break #to end the project click a
    #releases the cam  and all windowns 
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # to run the code endlessly
    while True:
        flag = input("\n1) Detect humans through External Cams (webcam)\n2) Detect humans through Prestored Video\n3) Motion Detection\n4) exit\nEnter your choice(1 or 2 or 3 or 4): ")

        if flag == "1":
            detect(0)  # Use 0 to indicate the default camera (webcam)
        elif flag == "2":
            path = input("Enter path with video name (eg: ./Sample videos/sample video.avi): ")
            detect(path)
        elif flag == "3":
            path = input("Enter path with video name (eg: '0' For Webcam and  './Sample videos/sample video.avi' for prerecorded videos): ")
            motion(0) if path == "0" else motion(path)
        elif flag == "4":
            break
        else:
            print("!!INVALID OPERATION!!, please enter valid operation (1 or 2 or 3)\n")