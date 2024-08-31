# Ref: https://www.youtube.com/watch?v=NGQgRH2_kq8&t=2s
# Camera changes, distance and width-height also change.
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone
import time

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
hf = 720
wf = 1280
cap.set(3, wf)  # set new width
cap.set(4, hf)  # set new height

# Crack image
crack_img = cv2.imread("GameResources/punch/crack.png", cv2.IMREAD_UNCHANGED)
# resize the effect image
if crack_img.shape[0] > 200:
    crack_img = cv2.resize(crack_img, ((crack_img.shape[1] * 200) // crack_img.shape[0], 200))  # (w,h)

# Setting white background
whiteBg = np.ones([hf, wf, 3], dtype=np.uint8)*(255,255,255)
whiteBg = whiteBg.astype(np.uint8)
# print(whiteBg)
Bg = whiteBg.copy()

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Find function to calculate actual distance from camera
# x is the raw distance calculated between landmarks 5 and 17
# y is the measured value in cm from camera
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
A, B, C = coff

# Game Variables
push = 0
score = 0
timeStart = time.time()
totalTime = 20
color = (255,0,255)

# Initial values
img_ratio = 0.2  # transparent level of camera frame (player)

# Loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if time.time()-timeStart < totalTime:

        hands, img = detector.findHands(img, draw=False)

        if hands:
            lmList = hands[0]['lmList']
            x, y, w, h = hands[0]['bbox']
            x1, y1 = lmList[5][0:2]
            x2, y2 = lmList[17][0:2]
            print(f"width: {w}, height: {h}")

            # distance in image between landmarks 5 and 17
            distance = int(math .sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))

            # actual distance in cm from camera
            distanceCM = A * distance ** 2 + B * distance + C
            # print(distanceCM, distance)

            if distanceCM < 40 and w > 180 and h > 180 and push == 0: # punch
                cvzone.overlayPNG(Bg, crack_img, (x, y))  # (x,y)
                push = 1
                score += 1
                color = (0,255,0)

            if distanceCM > 60 and w < 130 and h < 170 and push == 1: # release
                push = 0
                color = (255,0,255)
            print(f"push = {push}")
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10), colorR = color)

        # Time and score
        cvzone.putTextRect(img, f'Time: {int(totalTime-(time.time()-timeStart))}',
                           (1000, 75), scale=3, offset=20)
        cvzone.putTextRect(img, f'Score: {str(score).zfill(2)}', (60, 75), scale=3, offset=20)
    else:
        cv2.putText(Bg, 'Game Over', (400, 300), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=(0, 0, 255), thickness=12)
        cv2.putText(Bg, 'Game Over', (400, 300), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=(0, 0, 0), thickness=7)
        cv2.putText(Bg, f'Your Score: {score}', (390, 400), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(0, 0, 255), thickness=12)
        cv2.putText(Bg, f'Your Score: {score}', (390, 400), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(0, 0, 0), thickness=7)
        cv2.putText(Bg, 'Press R to restart', (390, 500), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 0, 255), thickness=5)
        img_ratio = 0

    imgOutput = cv2.addWeighted(img, img_ratio, Bg, 1 - img_ratio, 0)  # + 0
    cv2.imshow("Punch Crack", imgOutput)
    key = cv2.waitKey(1)

    if key == ord('r'):
        push = 0
        score = 0
        timeStart = time.time()
        Bg = whiteBg.copy()
        img_ratio = 0.2
    elif key == 27:  # Esc
        break