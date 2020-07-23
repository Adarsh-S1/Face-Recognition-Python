import cv2
import numpy as np
import face_recognition
import os
imgad=face_recognition.load_image_file('Adarsh.jpg')
imgad=cv2.cvtColor(imgad,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('Adarsh Test.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
