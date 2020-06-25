import os
import numpy as np
import cv2
cam = cv2.VideoCapture(0)

name = input("Enter name of person:")

path = 'images'
print(path)
directory = os.path.join(path, name)
print(directory)
if not os.path.exists(directory):
	os.makedirs(directory, exist_ok = 'True')

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0
while(True): #loop forever
    ret, img = cam.read()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 2)     
        count += 1
        #saving image in an name  folder
        # Save the captured image into the datasets folder
        cv2.imwrite(os.path.join(directory , str(name+str(count) + ".jpg" )), img[y1:y2,x1:x2])
        cv2.imshow('image', img) #window to show image
    k = cv2.waitKey(200) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
cam.release()
cv2.destroyAllWindows()
