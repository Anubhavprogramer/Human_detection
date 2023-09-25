import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


# creating credentials to database
cred = credentials.Certificate("ServiceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://humandetection-43fa7-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket':"humandetection-43fa7.appspot.com"
})


#importing the images of persons
folderPath = 'person'
peoplefolderpath = os.listdir(folderPath)
# print(peoplefolderpath)
imglist = []
personIDs =[]

# this loop stores the data into list
for path in peoplefolderpath:
    # add images to img list
    imglist.append(cv2.imread(os.path.join(folderPath,path)))
    # print(path)
    # print(os.path.splitext(path))
    # add id to id-list
    personIDs.append(os.path.splitext(path)[0])


    # uploading images to server
    filename = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob= bucket.blob(filename)
    blob.upload_from_filename(filename)

# print(len(imglist))
# print(personIDs)

# function to generate encodings
def findencodings(imageslist):
    # print(imageslist)
    encodelist=[]
    for img in imageslist:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        # print(len(encode))
        encodelist.append(encode)
    # retun the encoded list
    return encodelist



print("Encoding started.....")
# print(len(imglist))
# calling the encoding function
encodelistknown = findencodings(imglist)
encodelistknownwithIDs=[encodelistknown,personIDs]
# print(encodelistknown)
print("Encoding Completed.....")

# dumping the data into pickle file
file=open("encodefile.p",'wb')
pickle.dump(encodelistknownwithIDs,file)
file.close()
# pickle file closed

print("file saved....")