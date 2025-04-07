import numpy as np
import face_recognition
import os
import cv2 
from datetime import datetime
import csv


path = r'C:\Users\Prateek\OneDrive\Desktop\Face_Recognition_Project\Dataset'

images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#  Function to find face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#  Function to mark attendance safely
def markAttendance(name):
    filename = 'Attendance.csv'

    # Create file if it doesn't exist
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Time'])

    try:
        with open(filename, 'r+', newline='') as f:
            existingData = f.readlines()
            nameList = [line.split(',')[0] for line in existingData[1:]]  # Skip header

            if name not in nameList:
                now = datetime.now()
                timeStr = now.strftime('%H:%M:%S')
                f.write(f'{name},{timeStr}\n')
    except PermissionError:
        print("⚠️ Attendance.csv is open! Please close it and run again.")

# Encode known images
encodeListKnown = findEncodings(images)
print('✅ Face Encoding Complete!')

# 🎥 Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    success, img = video_capture.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDist)
        name_display = "Unknown"

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            name_display = name
            markAttendance(name)  # ✅ Mark attendance

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2 + 60, y2), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, name_display, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow(' Face Recognition Attendance', img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
