import cv2

trainedData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('group1.jpeg')

grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faceCoordinates = trainedData.detectMultiScale(grayimg)

for i in range(0,3):
    x,y,w,h = faceCoordinates[i]
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('group photo', img)
cv2.waitKey()

print('END OF PROGRAM')