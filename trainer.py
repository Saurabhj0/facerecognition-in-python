import os
import cv2
import numpy as np

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataset'

def getimageswithid(path):
     imagepaths_list=[os.path.join(path,f) for f in os.listdir(path)]
     faces=[]
     ids=[]
     print("Training the model")
     for imagepath in imagepaths_list:
          faceimg=cv2.imread(imagepath)
          faceimage_gray = cv2.cvtColor(faceimg, cv2.COLOR_BGR2GRAY)
          facenp=np.array(faceimage_gray,'uint8')
          ID=os.path.split(imagepath)[-1].split('_')[1]
          faces.append(facenp)
          ids.append(int(ID))
          #cv2.imshow("training",facenp)
          cv2.waitKey(10)
     return ids,faces

ids,faces=getimageswithid(path)
recognizer.train(faces,np.array(ids))
recognizer.save('recognizer/trainingdata.yml')
cv2.destroyAllWindows()
