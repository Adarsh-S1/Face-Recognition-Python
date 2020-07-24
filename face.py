import cv2
import numpy as np
import face_recognition

imgad = face_recognition.load_image_file('ImagesBasic/Adarsh.jpg')
imgad = cv2.cvtColor(imgad, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Adarsh Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgad)[0]
encodeAd = face_recognition.face_encodings(imgad)[0]
cv2.rectangle(imgad, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeAd], encodeTest)
faceDis = face_recognition.face_distance([encodeAd], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('ADARSH', imgad)
cv2.imshow('ADARSH TEST', imgTest)
cv2.waitKey(0)
