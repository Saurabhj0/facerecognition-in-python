import cv2
cam=cv2.VideoCapture(0)
face_detect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

enrollment=input("Enter the enrollment number")
sampleno=0
while True:
    ret,frame=cam.read()
    
    gray=cv2.cvtColor( frame,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        sampleno+=1
        cv2.imwrite("dataset/student_"+enrollment+"_"+str(sampleno)+".JPG",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(193,182,255),2)
        cv2.waitKey(50)
    cv2.imshow('face',frame)
    if sampleno>15 or cv2.waitKey(1) is "13":
        break
cam.release()
cv2.destroyAllWindows()


