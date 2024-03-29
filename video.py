import cv2

trainedData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture('clip.mp4')

while True:

    success, img = cam.read()

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceCoordinates = trainedData.detectMultiScale(grayimg)

    for x,y,w,h in faceCoordinates:
        cv2.rectangle(img, (x,y), (x+w, y+h), (200,0,256), 3)

    cv2.imshow('camera', img)

    key = cv2.waitKey(1)
    if(key==65 or key==97):
        break

print('END OF PROGRAM')
        