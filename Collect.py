import cv2
import urllib
import numpy as np

URL = 'http://192.168.43.1:8080/shot.jpg'
face_data = r'C:/Users/HP/projectnew/haarcascade_frontalface_alt.xml'
classifier = cv2.CascadeClassifier(face_data)
data1 = []
ret = True

while ret:
    img_url = urllib.request.urlopen(URL)
    image = np.array(bytearray(img_url.read()),np.uint8)
    frame = cv2.imdecode(image,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
    if faces is not None:
        for x,y,w,h in faces:
            
            
            face_image = frame[y:y+h,x:x+w].copy()
            
            if len(data1)<=100:
                 data1.append(face_image)
                 cv2.putText(frame,str(len(data1)),(200,200),
                            cv2.FONT_HERSHEY_COMPLEX,1,
                            (255,0,0),2)
            else:
                cv2.putText(frame,'complete',(200,200),
                            cv2.FONT_HERSHEY_COMPLEX,1,
                            (255,0,0),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)   
            
    cv2.imshow('capture',frame)
    if cv2.waitKey(1)==ord('q'):
            break
        

cv2.destroyAllWindows()

name = input("enter name : ")
c = 0
for i in data1:
    cv2.imwrite(r"C:/Users/HP/projectnew/images"+name+'_'+str(c)+'.jpg',i)
    c = c+1
