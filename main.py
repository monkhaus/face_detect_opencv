import cv2

# haar cascades xml files
face_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_eye.xml')
# image
img = cv2.imread('resources/face2.jpg')
# grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scaleFactor = 1.05
minNeighbors = 6

# faces
faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

    roi_gray = gray[y:y+h, x:x+w]
    roi_img = img[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
         cv2.rectangle(roi_img, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
         

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    

    
    

