import random
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

# Cockroach
cap_cockroach1 = cv2.VideoCapture('GameResources/cockroach/cockroach1.gif')
ret, frame_cockroach1 = cap_cockroach1.read()  # cockroach1
hc1_L = 130 # set height of cockroach1
wc1_L = (frame_cockroach1.shape[1] * hc1_L) // frame_cockroach1.shape[0]
cap_cockroach2_1 = cv2.VideoCapture('GameResources/cockroach/cockroach2_1.gif')
ret, frame_cockroach2 = cap_cockroach2_1.read()  # cockroach2
hc2_L = 250 # set height of cockroach2
wc2_L = (frame_cockroach2.shape[1] * hc2_L) // frame_cockroach2.shape[0]
cap_cockroach2_2 = cv2.VideoCapture('GameResources/cockroach/cockroach2_2.gif')
cap_cockroach2 = cap_cockroach2_1
# Cockroach 3-4 use original size.
cap_cockroach3 = cv2.VideoCapture('GameResources/cockroach/cockroach3.gif')
ret, frame_cockroach3 = cap_cockroach3.read()  # cockroach3
wc3, hc3 = frame_cockroach3.shape[1], frame_cockroach3.shape[0]
cap_cockroach4 = cv2.VideoCapture('GameResources/cockroach/cockroach4.gif')
ret, frame_cockroach4 = cap_cockroach4.read()  # cockroach4
wc4, hc4 = frame_cockroach4.shape[1], frame_cockroach4.shape[0]

# for random congratulation images when finish game
a = random.randint(0, 1)

# Setting white background
whiteBg = np.ones([hf, wf, 3], dtype=np.uint8)*(255,255,255)
whiteBg = whiteBg.astype(np.uint8)
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
open_w, open_h = 0, 0
color = (255, 0, 255)
counter = 0
score = 0
timeStart = time.time()
totalTime = 100
time_grasp = 0

# Initial values
img_ratio = 0.1  # transparent level of camera frame (player)
cx = random.randint(100, wf-wc1_L)
cy = random.randint(100, hf-(2*hc1_L))
speed = 50 # speed of cockroach2

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
            # print(f"width: {w}, height: {h}")

            # distance in image between landmarks 5 and 17
            distance = int(math .sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))

            # actual distance in cm from camera
            distanceCM = A * distance ** 2 + B * distance + C
            # print(distanceCM, distance)

            if distanceCM < 50: # move close to camera
                if x < cx+(0.3*wc1_L) < x + w and y < cy < y + h:  # grasp cockroach
                    counter = 1 # start to count time for grasping

            if distanceCM > 60:  # move far from camera
                counter = 0
                color = (255, 0, 255)
                time_grasp = 0

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10), colorR=color)

        if score < 3:
            ret, frame_cockroach1 = cap_cockroach1.read()  # cockroach1
            wc1, hc1 = wc1_L, hc1_L
        elif score == 10:
            if a:
                ret, frame_cockroach1 = cap_cockroach3.read()  # cockroach3
                wc1, hc1 = wc3, hc3
            else:
                ret, frame_cockroach1 = cap_cockroach4.read()  # cockroach4
                wc1, hc1 = wc4, hc4
            img_ratio = 0
        elif score >= 3:
            ret, frame_cockroach1 = cap_cockroach2.read()  # cockroach
            wc1, hc1 = wc2_L, hc2_L
            counter = 0

        if success and ret:
            Bg = whiteBg.copy()
            frame_cockroach_1 = cv2.resize(frame_cockroach1, (wc1, hc1))  # (w,h)
            # print(frame_cockroach1)
            _, mask_cockroach = cv2.threshold(frame_cockroach_1[:, :, 0], 245, 255, cv2.THRESH_BINARY_INV)
            frame_cockroach_dis = cv2.merge([frame_cockroach_1, mask_cockroach])
            if counter:
                print(f'start: {time_grasp}')
                if time_grasp == 1:
                    open_w, open_h = w, h # hand width and height of first frame of starting grasp
                time_grasp += 1
                color = (0, 255, 0)
                # print(w,0.7 *open_w,h,0.7*open_h)
                if w < 0.7*open_w and h < 0.7*open_h and time_grasp <= 50:  # wisp
                    print(f'grasp: {time_grasp}')
                    score += 1
                    counter = 0
                    open_w, open_h = 0, 0
                    cx = random.randint(100, wf - wc1)
                    cy = random.randint(100, hf - (2*hc1))
                    color = (255, 0, 255)
                    cap_cockroach1.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop gif animation
                if time_grasp >= 50: # time over for grasping
                    print(f'over: {time_grasp}')
                    cx = random.randint(100, wf - wc1)
                    cy = random.randint(100, hf - (2*hc1))
                    color = (255, 0, 255)
                    counter = 0
                    open_w, open_h = 0, 0
            else:
                color = (255, 0, 255)
                time_grasp = 0
                if score == 10:
                    if a:
                        cap_cockroach2 = cap_cockroach3 # cockroach3
                    else:
                        cap_cockroach2 = cap_cockroach4 # cockroach4
                    # cx = (wf - wc1) // 2
                    cy = np.clip(cy, 100, hf-hc1)
                elif score >= 3:
                    cx += -speed
                    # print(speed)
                    if cx < - (0.1*wc1) or cx > wf - wc1: # move out of left or right side
                        # print(cx)
                        speed = -speed
                        score -= 1
                        cy = random.randint(100, hf - (2*hc1))  # new y position
                        if cx < 0:
                            cap_cockroach2 = cap_cockroach2_2
                        else:
                            cap_cockroach2 = cap_cockroach2_1
                    if hands:
                        if hands[0]['type'] == "Right" and x+w-(1.4*wc1) < cx <= x+w-(0.4*wc1) and speed > 0: # cockroach moves from right
                            # print(cx)
                            score += 1
                            speed = -speed
                            cap_cockroach2 = cap_cockroach2_2
                        elif hands[0]['type'] == "Left" and x-(0.6*wc1) <= cx < x and speed < 0: # cockroach moves from left
                            # print(cx)
                            score += 1
                            speed = -speed
                            cap_cockroach2 = cap_cockroach2_1

            Bg = cvzone.overlayPNG(Bg, frame_cockroach_dis, (cx,cy))  # (x,y)
            buffer_frame = frame_cockroach_dis.copy()
        else:
            cap_cockroach1.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop gif animation
            cap_cockroach2_1.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop gif animation
            cap_cockroach2_2.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop gif animation
            cap_cockroach3.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop gif animation
            cap_cockroach4.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop gif animation
            Bg = cvzone.overlayPNG(Bg, buffer_frame, (cx,cy))  # (x,y)

        # Time and score
        if score == 10:
            cvzone.putTextRect(Bg, f'TimeU: {totalTime-timeLeft}',  # time used
                           (950, 75), scale=3, offset=20)
        elif score < 10:
            timeLeft = int(totalTime - (time.time() - timeStart))   # time left
            cvzone.putTextRect(Bg, f'TimeL: {timeLeft}',
                           (950, 75), scale=3, offset=20)

        cvzone.putTextRect(Bg, f'Score: {str(score).zfill(2)}', (60, 75), scale=3, offset=20)
    else:
        cvzone.putTextRect(Bg, 'Game Over', (400, 400), scale=5, offset=30, thickness=7)
        cvzone.putTextRect(Bg, f'Your Score: {score}', (450, 500), scale=3, offset=20)
        cvzone.putTextRect(Bg, 'Press R to restart', (460, 575), scale=2, offset=10)

    imgOutput = cv2.addWeighted(img, img_ratio, Bg, 1 - img_ratio, 0)  # + 0

    cv2.imshow("Grasp Cockroach", imgOutput)
    key = cv2.waitKey(1)

    if key == ord('r'):
        timeStart = time.time()
        score = 0
        img_ratio = 0.1
        a = random.randint(0, 1)
        cap_cockroach2 = cap_cockroach2_1
        speed = 50
        cx = random.randint(100, wf - hc1_L)
        cy = random.randint(100, hf-(2*hc1_L))
        cap_cockroach1.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop gif animation
    elif key == 27:  # Esc
        break