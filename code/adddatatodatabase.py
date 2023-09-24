import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("ServiceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://humandetection-43fa7-default-rtdb.asia-southeast1.firebasedatabase.app/"
})


ref = db.reference('data_of_people')
# creating data to be added
data={
    "123654":
    {
        "name":"Swastik",
        "LastDetectedTime":"2023-09-23 00:00:00",
        "reason":"Wanted"
    },
    "987654":
    {
        "name":"Avanish",
        "LastDetectedTime":"2023-09-23 00:00:00",
        "reason":"Wanted"
    },
    "951375":
    {
        "name":"Srishti",
        "LastDetectedTime":"2023-09-23 00:00:00",
        "reason":"Wanted"
    },
    "147896":
    {
        "name":"Piyush",
        "LastDetectedTime":"2023-09-23 00:00:00",
        "reason":"Wanted"
    },
    "951753":
    {
        "name":"Anubhav",
        "LastDetectedTime":"2023-09-23 00:00:00",
        "reason":"Wanted"
    },
    "987321":
    {
        "name":"Alka",
        "LastDetectedTime":"2023-09-23 00:00:00",
        "reason":"Wanted"
    },
}
# uploading the data
for key, value in data.items():
    ref.child(key).set(value)