import cv2

trainedData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcame = cv2.VideoCapture(0)

while True:

    success, img = webcame.read()

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceCoordinates = trainedData.detectMultiScale(grayimg)

    for x, y, w, h in faceCoordinates:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 256), 2)

    cv2.imshow('camera', img)
    key = cv2.waitKey(1)
    if(key ==65 or key==97):
        break

webcame.release()

print('END OF PROGRAM')