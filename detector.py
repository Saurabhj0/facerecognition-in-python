import cv2
cam=cv2.VideoCapture(0)
face_detect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\trainingdata.yml")
id=0
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,frame=cam.read()
    gray=cv2.cvtColor( frame,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(193,182,255),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        cv2.putText(frame,str("TCA"+str(id)),(x,y+h),font,1,255)
    cv2.imshow('face',frame)
    if cv2.waitKey(1)==13:
        break
cam.release()
cv2.destroyAllWindows()
