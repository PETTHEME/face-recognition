import cv2
import os
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
face_capture = cv2.VideoCapture(0)
while True:
    # capture the frames
    ret, frames = face_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # make regtangle around your face
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # showing the frames
    cv2.imshow('face reg', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_capture.release()
cv2.destroyAllWindows()